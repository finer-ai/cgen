import requests
import base64
import os
from pathlib import Path
import argparse
import time
import json

def save_images(images: list, output_dir: Path, prefix: str = "generated", subfolder: str = None):
    """Base64エンコードされた画像リストを保存する"""
    # タイムスタンプ付きのディレクトリ名を作成
    output_dir = output_dir / subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    
    for i, img_base64 in enumerate(images):
        img_data = base64.b64decode(img_base64)
        output_path = output_dir / f"{prefix}_{i}.png"
        
        with open(output_path, "wb") as f:
            f.write(img_data)
        saved_paths.append(output_path)
    
    return saved_paths

def call_runpod_endpoint(prompt: str, endpoint_id: str, api_key: str, **kwargs):
    """RunPodエンドポイントを呼び出す"""
    start_time = time.time()
    
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    
    # リクエストの構築
    request = {
        'input': {
            'prompt': prompt,
            'tag_candidate_generation_template': kwargs.get('tag_candidate_generation_template', None),
            'tag_normalization_template': kwargs.get('tag_normalization_template', None),
            'tag_filter_template': kwargs.get('tag_filter_template', None),
            'tag_weight_template': kwargs.get('tag_weight_template', None),
            'negative_prompt': kwargs.get('negative_prompt', ''),
            'num_images': kwargs.get('num_images', 1),
            'steps': kwargs.get('steps', 30),
            'cfg_scale': kwargs.get('cfg_scale', 7.0),
            'width': kwargs.get('width', 512),
            'height': kwargs.get('height', 768)
        }
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.post(url, json=request, headers=headers)
        response.raise_for_status()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"API call took: {elapsed_time:.2f} seconds")

        data = response.json()
        
        if "error" in data:
            print(f"Error: {data['error']}")
            return None

        # 生成された画像を保存
        output_dir = Path("output")
        print(len(data["output"]["images"]))
        print(len(data["output"]["bodylines"]))

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        saved_paths = save_images(
            data["output"]["images"],
            output_dir,
            prefix="generated",
            subfolder=timestamp
        )
        # ボディラインを保存
        saved_paths = save_images(
            data["output"]["bodylines"],
            output_dir,
            prefix="bodyline",
            subfolder=timestamp
        )
        
        # 生成されたタグとパラメータを保存
        metadata = {
            "generated_tags": data["output"]["generated_tags"],
            "parameters": data["output"]["parameters"]
        }
        
        metadata_path = output_dir / timestamp / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        print(f"Generated images saved to: {[str(p) for p in saved_paths]}")
        print(f"Metadata saved to: {metadata_path}")
        
        return saved_paths, metadata
        
    except requests.exceptions.RequestException as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"API call took: {elapsed_time:.2f} seconds")
        print(f"Error calling RunPod endpoint: {e}")
        return None
    
templates = {}
templates["tag_candidate_generation_template"] = """
Generate Danbooru tags from the prompt. The prompt may either be a question or instruction requesting a scene to be drawn, or a direct description of the desired scene without a question.
First, extract the scene. Then, generate up to 20 Danbooru tags, separated by commas, detailing all relevant aspects including character count, group tags, and other necessary details.

Important character count rules:
- Unless explicitly specified otherwise in the prompt, assume there is only one character.
- For a single character, use exactly one character tag (e.g., '1girl', '1boy', or '1other') along with 'solo'.
- For multiple characters, use tags that match the exact number of characters (e.g., '2girls', '2boys', 'multiple_girls').
- When gender is not explicitly specified in the prompt, prioritize using '1girl' over '1boy'.

Additional rules:
- If there is a single human or humanoid character (including orcs, elves, etc.), use 'solo' along with '1boy' or '1girl'.
- If there is a single non-humanoid character, use 'solo' along with '1other'.
- If applicable, combine species tags (e.g., 'orc', 'elf') with character count tags.

Prompt: {prompt}
Tags (up to 20):
"""

templates["tag_normalization_template"] = """
You are a Danbooru tag matcher. Find exact matches in the context for the input.
Rules:
1. Return only exact matches from the context.
2. Check main tags and "Other names".
3. If no match, return input as is.
4. Provide one tag output per input.
5. Match tag names or other names, not descriptions.
6. For character counts and groups:
- Use 'solo' for a single character (human, humanoid, or non-humanoid)
- Use '1boy' for a single male character or humanoid male-presenting character (including orcs, elves, etc.)
- Use '1girl' for a single female character or humanoid female-presenting character
- Use '1other' for a single non-humanoid character
- For other character counts, use appropriate tags like '2boys', 'multiple_girls', etc.
7. Prioritize character count and group tags over other matches if applicable.
8. When using '1boy', '1girl', or '1other', also include the 'solo' tag.
9. Do not use square brackets or double square brackets around tags.
10. Match all relevant tags, including those for background, clothing, expressions, and actions.
11. For humanoid characters like orcs, elves, etc., use appropriate character count tags ('1boy', '1girl', etc.) along with their species tag.

Context: {context}
Input: {query}
Output:
"""

templates["tag_filter_template"] = """
From the tag list below, please remove tags that significantly deviate from the given context,
and keep only the tags that are relevant.
Please output the filtered tags as a comma-separated list.
Note that the context represents the content we want to include in the prompt, so please keep the tags that align with the context.

# Example Start ####
Example Context: 女の子がジャンプしているポーズを描いてください。セーラー服ではなくブレザーを着ている。
Example Tag List: original, 1girl, jumping, embarrassed, jacket, solo, outdoors, day, long hair, skirt, kneehighs, smile, looking at viewer, miniskirt, monochrome, monster, open clothes, open jacket, open mouth, pleated skirt, shoes, socks, teeth, upper teeth only, wide shot, wind, wind lift
Example Output: original, 1girl, jumping, jacket, solo, long hair, skirt, smile, looking at viewer, miniskirt, open jacket, shoes, socks
# Example End ####
Note: (deleted tags: embarrassed, outdoors, day, kneehighs, monochrome, monster, open clothes, open mouth, pleated skirt, teeth, upper teeth only, wide shot, wind, wind lift)

Context: {context_prompt}

Tag List: {tags_str}

Output:"""

templates["tag_weight_template"] = """
Please analyze the context and add weights ONLY when there are explicit or strongly implied modifiers in the context.
Be aggressive in applying weights when modifiers are present, but DO NOT add weights when there are no modifiers.

Important rules for weight application:
1. NO WEIGHTS by default:
   - If a tag has no direct or implied modifier, leave it as is without weights
   - Do not add weights based on assumptions or general knowledge
   - Only add weights when there is clear evidence in the context

2. POSE-RELATED weights (:1.8):
   - Apply :1.8 to ALL pose-contributing tags when the context specifically requests a pose
   - This includes:
     a) Direct pose tags: standing, sitting, jumping, running, lying, kneeling, etc.
     b) Body part position tags: arms_up, crossed_legs, spread_arms, hands_in_pockets, etc.
     c) Action-implying tags: looking_back, hair_flip, stretching, twirling, etc.
     d) Pose-affecting clothing tags: skirt_flip, wind_lift, floating_hair, etc.
     e) Dynamic elements: wind, motion_blur, action_lines, etc.

3. Direct modifiers - Apply to ALL related tags when present:
   - Low intensity (:0.3): ちょっと (a little), 少し (slightly), やや (somewhat), 微妙に (subtly), 控えめに (moderately)
   - High intensity (:1.7): すごく (very), とても (extremely), 非常に (highly), かなり (considerably), めちゃくちゃ (incredibly)

4. Mood/Atmosphere modifiers - Apply ONLY when explicitly mentioned or strongly implied:
   When the context explicitly mentions these moods, apply weights to ALL related tags:
   
   a) Suggestive/Erotic mood:
      - If words like エッチ, セクシー, 色気 appear:
        * Apply :0.3 to: sexually_suggestive, revealing_clothes, suggestive_pose, etc.
        * Also apply :0.3 to related tags: thighhighs, miniskirt, bare_shoulders, etc.
   
   b) Cute/Innocent mood:
      - If words like 可愛い, innocent appear:
        * Apply :1.7 to: cute, innocent, pure, etc.
        * Also apply :1.7 to related tags: smile, flower, pastel_colors, etc.
   
   c) Intensity modifiers stack with mood:
      - "ちょっとエッチ" = (sexually_suggestive:0.3)
      - "すごくエッチ" = (sexually_suggestive:1.7)

5. Context Analysis - Strict Rules:
   - Only consider explicit modifiers or very strong contextual implications
   - Do not add weights based on subtle implications or assumptions
   - When in doubt, leave the tag without weights
   - Pose weights (:1.8) can stack with mood/modifier weights

# Examples ####
Example 1 (Pose with modifiers):
Context: ちょっとエッチな感じで女の子がジャンプしているポーズを描いてください。スカートがふわっとなっています。
Tags: original, 1girl, jumping, solo, sexually_suggestive, skirt, smile, thighhighs, wind_lift
Output: original, 1girl, (jumping:1.8), solo, (sexually_suggestive:0.3), (skirt:0.3), smile, (thighhighs:0.3), (wind_lift:1.8)

Example 2 (Simple pose):
Context: 女の子が両手を広げてバランスを取っているポーズを描いてください。
Tags: original, 1girl, standing, spread_arms, balancing, skirt
Output: original, 1girl, (standing:1.8), (spread_arms:1.8), (balancing:1.8), skirt

Example 3 (Mixed modifiers with pose):
Context: すごく可愛らしい女の子が、ちょっとセクシーな感じで後ろを振り返るポーズ。
Tags: original, 1girl, looking_back, turning, cute, sexually_suggestive, smile
Output: original, 1girl, (looking_back:1.8), (turning:1.8), (cute:1.7), (sexually_suggestive:0.3), (smile:1.7)

Example 4 (Dynamic pose):
Context: 風で制服がなびいている中、ジャンプしている女の子を描いてください。
Tags: original, 1girl, jumping, school_uniform, wind, floating_hair, skirt_flip
Output: original, 1girl, (jumping:1.8), school_uniform, (wind:1.8), (floating_hair:1.8), (skirt_flip:1.8)

Example 5 (No clear modifiers):
Context: 放課後の教室で本を読んでいる女の子を描いてください。
Tags: original, 1girl, classroom, reading, book, school_uniform, afternoon
Output: original, 1girl, classroom, (reading:1.8), book, school_uniform, afternoon

Context: {context_prompt}

Tags: {tags_str}

Output:"""

def main():    
    parser = argparse.ArgumentParser(description='Call RunPod endpoint to generate images from prompt')
    parser.add_argument('--prompt', help='Text prompt for image generation')
    parser.add_argument('--negative-prompt', help='Negative prompt for image generation')
    parser.add_argument('--num-images', type=int, help='Number of images to generate')
    parser.add_argument('--steps', type=int, help='Number of generation steps')
    parser.add_argument('--cfg-scale', type=float, help='CFG scale')
    parser.add_argument('--width', type=int, help='Image width')
    parser.add_argument('--height', type=int, help='Image height')
    parser.add_argument('--endpoint-id', help='RunPod endpoint ID')
    parser.add_argument('--api-key', help='RunPod API key')
    parser.add_argument('--tag-candidate-generation-template', help='Template for tag candidate generation')
    parser.add_argument('--tag-normalization-template', help='Template for tag normalization')
    parser.add_argument('--tag-filter-template', help='Template for tag filtering')
    parser.add_argument('--tag-weight-template', help='Template for tag weighting')
    
    args = parser.parse_args()
    
    import dotenv
    dotenv.load_dotenv()
      
    prompt = args.prompt or "ジャンプをしている女の子のポーズ"
    # prompt = args.prompt or "壁に寄りかかりながらピースサインをしている女の子のポーズ"
    # prompt = args.prompt or "prompt:original, 1girl, solo, (jumping:1.8), jacket, school uniform, pose, miniskirt, brown hair, blue eyes, pleated skirt, red footwear, red jacket, shoes, striped clothes, thighs, white thighhighs, expression"
    endpoint_id = args.endpoint_id or os.getenv("RUNPOD_ENDPOINT_ID")
    api_key = args.api_key or os.getenv("RUNPOD_API_KEY")
    negative_prompt = args.negative_prompt or "nsfw, sensitive, from behind, lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, missing arms, extra arms, missing legs, extra legs, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry"
    num_images = args.num_images or 4
    steps = args.steps or 30
    cfg_scale = args.cfg_scale or 10.0
    width = args.width or 832
    height = args.height or 1216

    tag_candidate_generation_template = args.tag_candidate_generation_template or templates["tag_candidate_generation_template"]
    tag_normalization_template = args.tag_normalization_template or templates["tag_normalization_template"]
    tag_filter_template = args.tag_filter_template or templates["tag_filter_template"]
    tag_weight_template = args.tag_weight_template or templates["tag_weight_template"]
    
    result = call_runpod_endpoint(
        prompt,
        endpoint_id,
        api_key,
        tag_candidate_generation_template=tag_candidate_generation_template,
        tag_normalization_template=tag_normalization_template,
        tag_filter_template=tag_filter_template,
        tag_weight_template=tag_weight_template,
        negative_prompt=negative_prompt,
        num_images=num_images,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height
    )
    
    if result:
        print("Processing completed successfully")
        
# python call.py "図書館で本を読んでいる女の子。眼鏡をかけていて、真面目そうな見た目なんですが、なんとなく大人っぽい魅力があふれています。" --endpoint-id qidc047mo1cano --api-key rpa_XWRZGKA8089VRA02YEZU574GH9MA5QTTPLY4LPKW1sjqle

if __name__ == "__main__":
    main() 
