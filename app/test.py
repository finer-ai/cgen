import asyncio
from services.rag_service import RAGService
from services.dart_service import DartService
async def main():
    print()
    # RAGサービス
    rag_service = RAGService()

    prompt_ja = "ベッドの上で横向きに寝そべっている金髪の女の子"
    # prompt_en = "A girl with blonde hair lying on her bed, facing sideways"
    keywords = await rag_service.extract_keywords(prompt_ja)
    print(keywords, "\n")

    # キーワードから関連文脈を取得
    # keywords = ["""A portrait of an artist drawn by the artist themselves, this also includes an [[original]] character/[[virtual youtuber]] representing them."""]
    # tags = await rag_service.retrieve_tags([prompt_en])
    # print([tag.split('\n') for tag in tags])

    # Dartサービス
    dart_service = DartService()
    # some_tags = ["1girl", "solo", "bed", "ass", "sitting"]
    # タグ候補を生成
    final_tags = await dart_service.generate_final_tags(keywords)
    print('final_tags_list', final_tags)
    print(', '.join(final_tags))
    print()

if __name__ == "__main__":
    asyncio.run(main())