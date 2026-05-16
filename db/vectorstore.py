"""
Vector Store 모듈
- PDF 파일 로드 → 텍스트 분할 → 임베딩 → ChromaDB 저장
- 유사도 검색 기능 제공
"""
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ─── 경로 설정 ───
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

# ─── 임베딩 모델 설정 ───
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"


def _get_embeddings():
    """HuggingFace 임베딩 모델 인스턴스를 반환합니다."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore():
    """
    data/pdfs/ 폴더의 모든 PDF 파일을 읽어 Vector DB를 구축합니다.
    
    Returns:
        Chroma: 구축된 Vector Store 인스턴스
    """
    # PDF 파일 목록 가져오기
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f"⚠️  {PDF_DIR} 폴더에 PDF 파일이 없습니다.")
        print("   PDF 파일을 넣은 후 다시 실행해주세요.")
        return None

    print(f"📄 {len(pdf_files)}개의 PDF 파일을 발견했습니다.")

    # 모든 PDF에서 문서 로드
    all_documents = []
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"   📖 로딩 중: {filename}")
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            # 메타데이터에 파일명 추가
            for doc in documents:
                doc.metadata["source_file"] = filename
            all_documents.extend(documents)
        except Exception as e:
            print(f"   ❌ {filename} 로드 실패: {e}")

    if not all_documents:
        print("❌ 로드된 문서가 없습니다.")
        return None

    print(f"📝 총 {len(all_documents)}페이지 로드 완료")

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"✂️  {len(chunks)}개 청크로 분할 완료")

    # 임베딩 & ChromaDB 저장
    print(f"🔄 임베딩 생성 중... (첫 실행 시 모델 다운로드에 시간이 걸릴 수 있습니다)")
    embeddings = _get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print(f"✅ Vector DB 구축 완료! ({CHROMA_DIR})")
    print(f"   저장된 청크 수: {len(chunks)}")
    return vectorstore


def get_vectorstore():
    """
    기존에 구축된 Vector DB를 로드합니다.
    구축된 DB가 없으면 None을 반환합니다.
    
    Returns:
        Chroma or None: Vector Store 인스턴스
    """
    if not os.path.exists(CHROMA_DIR):
        return None

    embeddings = _get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    # 실제 데이터가 있는지 확인
    try:
        collection = vectorstore._collection
        count = collection.count()
        if count == 0:
            return None
        return vectorstore
    except Exception:
        return None


def search_products(query: str, k: int = 3):
    """
    Vector DB에서 질문과 관련된 상품 정보를 검색합니다.
    
    Args:
        query: 검색할 질문 텍스트
        k: 반환할 문서 수
    
    Returns:
        list: 관련 문서 리스트 (Vector DB가 없으면 빈 리스트)
    """
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return []

    results = vectorstore.similarity_search(query, k=k)
    return results
