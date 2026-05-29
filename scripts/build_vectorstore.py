"""
Vector DB 빌드 스크립트
- data/pdfs/ 폴더에 있는 PDF 파일들을 읽어 ChromaDB에 저장합니다.

PDF 파일을 추가/변경한 후 이 스크립트를 다시 실행하면 DB가 재구축됩니다.
"""
import sys
import os
import shutil

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from db.vectorstore import build_vectorstore, CHROMA_DIR


def main():
    print("=" * 60)
    print("🔧 Vector DB 빌드 스크립트")
    print("=" * 60)

    # 기존 DB가 있으면 삭제 후 재구축
    if os.path.exists(CHROMA_DIR):
        print(f"\n⚠️  기존 Vector DB를 삭제합니다: {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)

    print("\n📦 Vector DB 구축을 시작합니다...\n")
    vectorstore = build_vectorstore()

    if vectorstore:
        print("\n" + "=" * 60)
        print("🎉 Vector DB 구축이 완료되었습니다!")
        print("   이제 streamlit run app.py 로 앱을 실행하세요.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Vector DB 구축에 실패했습니다.")
        print("   data/pdfs/ 폴더에 PDF 파일을 넣고 다시 시도해주세요.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
