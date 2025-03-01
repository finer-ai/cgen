import asyncio
from services.rag_service import RAGService
from services.dart_service import DartService
from utils.llm_utils import load_llm

def get_test_tag_candidates():
    tag_candidates = "1girl, solo, jumping, school uniform, jacket, posture, action figure, outdoors, day, long hair, brown hair, green eyes, smile, happy, energy, 1boy, cutesexyrobutts, kawaii"
    tag_candidates = tag_candidates.split(', ')
    return tag_candidates
    
def get_test_final_tags():
    final_tags = "original, 1girl, solo focus, jumping, outdoors, action, pose, brown hair, solo, 1boy, energy, facing away, fantasy, from behind, glowing, holding, holding sword, holding weapon, japanese clothes, katana, kimono, long hair, long sleeves, monster, obi, ponytail, red kimono, sandals, sash, sword, tabi, weapon, white legwear, wide sleeves, zouri"
    final_tags = final_tags.split(', ')
    return final_tags

async def test_service():
    llm = load_llm(use_local_llm=False)
    
    rag_service = RAGService(llm)
    # prompt = "「女の子がジャンプしているポーズ」を描いてください。セーラー服ではなくブレザーを着ている。ちょっとエッチな感じで。"
    prompt = "図書館で本を読んでいる女の子。眼鏡をかけていて、真面目そうな見た目なんですが、なんとなく大人っぽい魅力があふれています。"
    tag_candidates = await rag_service.generate_tag_candidates(prompt)
    # tag_candidates = get_test_tag_candidates()
    # print('##tag_candidates##\n', ', '.join(tag_candidates), "\n")

    dart_service = DartService(llm)
    final_tags = await dart_service.generate_final_tags(tag_candidates)
    # final_tags = get_test_final_tags()
    # print('##final_tags##\n', ', '.join(final_tags), "\n")

    filtered_tags = await dart_service.filter_tags_by_context(
        tags_str=", ".join(final_tags),
        context_prompt=prompt
    )
    print('##filtered_tags##\n', ', '.join(filtered_tags))
    # print('##deleted_tags##\n', ', '.join(set(final_tags) - set(filtered_tags)), "\n")
    # print('final_tagsに含まれているが、filtered_tagsに含まれていないタグは以下です。')
    # for tag in set(filtered_tags):
    #     if tag not in final_tags:
    #         print(tag)

if __name__ == "__main__":
    asyncio.run(test_service())