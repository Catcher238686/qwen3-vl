import json


def convert_to_qwen3vl_format(input_data, output_file):
    """
    将你的caption数据集转换为Qwen3-VL训练格式
    
    Args:
        input_data: 输入数据列表，每个元素包含 'image_path' 和 'caption'
        output_file: 输出JSONL文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in input_data:
            qwen_entry = {
                "image": item["image_path"],
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\n请为这张图片生成一个详细的描述。",
                    },
                    {
                        "from": "gpt",
                        "value": item["caption"]
                    }
                ]
            }
            f.write(json.dumps(qwen_entry, ensure_ascii=False) + '\n')


def convert_multiple_images_to_qwen3vl_format(input_data, output_file):
    """
    将多图caption数据集（如变化描述）转换为Qwen3-VL训练格式
    
    Args:
        input_data: 输入数据列表，每个元素包含 'images' 和 'caption'
        output_file: 输出JSONL文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in input_data:
            num_images = len(item["images"])
            image_tags = "\n".join(["<image>" for _ in range(num_images)])
            
            qwen_entry = {
                "images": item["images"],
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{image_tags}\n请描述这些图片的内容。",
                    },
                    {
                        "from": "gpt",
                        "value": item["caption"]
                    }
                ]
            }
            f.write(json.dumps(qwen_entry, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert caption dataset to Qwen3-VL training format")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--multi-image", action="store_true", help="Use multi-image format")
    
    args = parser.parse_args()
    
    with open(args.input, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    if args.multi_image:
        convert_multiple_images_to_qwen3vl_format(input_data, args.output)
    else:
        convert_to_qwen3vl_format(input_data, args.output)
    
    print(f"Conversion complete! Output saved to {args.output}")
