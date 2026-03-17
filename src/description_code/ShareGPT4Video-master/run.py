import argparse
import os
import tqdm
import pandas as pd
import numpy as np
import torch
from PIL import Image
from glob import glob

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (get_model_name_from_path, process_images,
                            tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def create_frame_grid(img_array, interval_width=50):
    n, h, w, c = img_array.shape
    grid_size = int(np.ceil(np.sqrt(n)))

    horizontal_band = np.ones((h, interval_width, c),
                              dtype=img_array.dtype) * 255
    vertical_band = np.ones((interval_width, w + (grid_size - 1)
                            * (w + interval_width), c), dtype=img_array.dtype) * 255

    rows = []
    for i in range(grid_size):
        row_frames = []
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < n:
                frame = img_array[idx]
            else:
                frame = np.ones_like(img_array[0]) * 255
            if j > 0:
                row_frames.append(horizontal_band)
            row_frames.append(frame)
        combined_row = np.concatenate(row_frames, axis=1)
        if i > 0:
            rows.append(vertical_band)
        rows.append(combined_row)

    final_grid = np.concatenate(rows, axis=0)
    return final_grid


def resize_image_grid(image, max_length=1920):
    width, height = image.size
    if max(width, height) > max_length:
        if width > height:
            scale = max_length / width
        else:
            scale = max_length / height

        new_width = int(width * scale)
        new_height = int(height * scale)

        img_resized = image.resize((new_width, new_height), Image.BILINEAR)
    else:
        img_resized = image
    return img_resized


def video_answer(prompt, model, processor, tokenizer, img_grid, do_sample=True,
                 max_new_tokens=200, num_beams=1, top_p=0.9,
                 temperature=1.0, print_res=False, **kwargs):
    if not isinstance(img_grid, (list, tuple)):
        img_grid = [img_grid]
    image_size = img_grid[0].size
    image_tensor = process_images(img_grid, processor, model.config)[0]
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).to(
        device=model.device, non_blocking=True)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token is not None else tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(
                dtype=torch.float16, device=model.device, non_blocking=True),
            image_sizes=[image_size],
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            use_cache=True,
            **kwargs)
        outputs = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)[0].strip()
    if print_res:  # debug usage
        print('### PROMPTING LM WITH: ', prompt)
        print('### LM OUTPUT TEXT:  ', outputs)

    return outputs


def single_test(model, processor, tokenizer, vid_path, qs, pre_query_prompt=None,  num_frames=16, conv_mode="plain"):
    def get_index(num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets

    def load_video(video_path, num_segments=8, return_msg=False):
        image_paths = sorted(glob(os.path.join(video_path, "*.jpg")))

        num_frames = len(image_paths)
        if num_frames == 0:
            print(video_path)
            raise ValueError("No images found in the directory!")

        # Get sampled frame indices
        frame_indices = get_index(num_frames, num_segments)

        # Load selected frames as NumPy arrays
        img_array = np.array([np.array(Image.open(image_paths[i])) for i in frame_indices])

        # Create frame grid
        img_grid = create_frame_grid(img_array, 50)
        img_grid = Image.fromarray(img_grid).convert("RGB")
        img_grid = resize_image_grid(img_grid)

        if return_msg:
            sec = ", ".join([str(i) for i in frame_indices])
            msg = f"The directory contains {len(frame_indices)} sampled frames at indices: {sec}."
            return img_grid, msg
        else:
            return img_grid
    if num_frames != 0:
        vid, msg = load_video(
            vid_path, num_segments=num_frames, return_msg=True)
    else:
        vid, msg = None, 'num_frames is 0, not inputing image'
    img_grid = vid
    conv = conv_templates[conv_mode].copy()
    if pre_query_prompt is not None:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + pre_query_prompt + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    llm_response = video_answer(prompt, model=model, processor=processor, tokenizer=tokenizer,
                                do_sample=False, img_grid=img_grid, max_new_tokens=512, print_res=False)
    return llm_response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="Lin-Chen/sharegpt4video-8b")
    parser.add_argument("--folder-path", type=str,
                        default="AnimalKingdom/action_recognition/dataset/image/")
    parser.add_argument("--output_csv", type=str, default="sharegpt_mammalnet.csv")
    parser.add_argument("--conv-mode", type=str, default="llava_llama_3")
    parser.add_argument("--query", type=str,
                        default="Provide a one-sentence caption that highlights the main animal actions in this video.")
    args = parser.parse_args()
    num_frames = 16
    pre_query_prompt = "The provided image arranges keyframes from a video in a grid view, keyframes are separated with white bands. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the frames."

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path, None, model_name, device_map='cpu' if not torch.cuda.is_available() else 'cuda')
    model = model.cuda().eval()

    general_path = args.folder_path
    videos_list = list()
    caption_list = list()

    for i, video_dir in enumerate(os.listdir(general_path)):
        if video_dir != '.DS_Store':
            outputs = single_test(model,
                                  processor,
                                  tokenizer,
                                  os.path.join(general_path, video_dir),
                                  qs=args.query,
                                  pre_query_prompt=pre_query_prompt,
                                  num_frames=num_frames,
                                  conv_mode=args.conv_mode)
            videos_list.append(video_dir)
            caption_list.append(outputs)
            if i % 500 == 0:
                df = pd.DataFrame({'video': videos_list, 'caption': caption_list})
                df.to_csv(args.output_csv, index=False)
    df = pd.DataFrame({'video': videos_list, 'caption': caption_list})
    df.to_csv(args.output_csv, index=False)

