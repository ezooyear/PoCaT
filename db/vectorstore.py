"""
Vector Store 모듈
- PDF 파일 로드 → 부모-자식 청크 분할 → 임베딩 → ChromaDB 저장
- BM25 하이브리드 검색 및 Cross-Encoder 리랭킹(BAAI/bge-reranker-v2-m3) 제공
"""
import os
import glob
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# ─── 경로 설정 ───
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

# ─── 임베딩 및 리랭커 모델 설정 ───
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# 리랭커 모델 싱글톤 캐시
_RERANKER_INSTANCE = None


def _get_embeddings():
    """HuggingFace 임베딩 모델 인스턴스를 반환합니다."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_reranker() -> CrossEncoder:
    """BAAI/bge-reranker-v2-m3 리랭커 싱글톤 인스턴스를 반환합니다."""
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None:
        print(f"🔄 리랭커 모델 로드 중... ({RERANKER_MODEL_NAME})")
        _RERANKER_INSTANCE = CrossEncoder(RERANKER_MODEL_NAME, device="cpu")
        print("✅ 리랭커 모델 로드 완료!")
    return _RERANKER_INSTANCE


def build_vectorstore():
    """
    data/pdfs/ 폴더의 모든 PDF 파일을 읽어
    부모-자식 청크 분할 기법으로 Vector DB를 구축합니다.
    """
    # PDF 파일 목록 가져오기
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f"⚠️  {PDF_DIR} 폴더에 PDF 파일이 없습니다.")
        print("   PDF 파일을 넣은 후 다시 실행해주세요.")
        return None

    print(f"📄 {len(pdf_files)}개의 PDF 파일을 발견했습니다.")

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

    # 1단계: 부모 청크 분할 (상세 컨텍스트 유지용 - 1000자)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    parent_docs = parent_splitter.split_documents(all_documents)
    print(f"✂️  부모 청크 분할 완료: {len(parent_docs)}개")

    # 2단계: 자식 청크 분할 (검색 정확도 향상용 - 250자)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    child_docs = []
    for i, p_doc in enumerate(parent_docs):
        # 부모 청크를 더 작게 분할
        sub_docs = child_splitter.split_documents([p_doc])
        for s_doc in sub_docs:
            # 자식 청크의 메타데이터에 부모 본문 및 부모 인덱스 ID를 매핑 저장
            new_metadata = dict(s_doc.metadata)
            new_metadata["parent_content"] = p_doc.page_content
            new_metadata["parent_id"] = i
            s_doc.metadata = new_metadata
            child_docs.append(s_doc)

    print(f"✂️  자식 청크 분할 완료: {len(child_docs)}개")

    # 기존 DB가 있다면 삭제하고 초기화
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("🧹 기존 Vector DB 디렉토리 삭제 완료")

    # 임베딩 & ChromaDB 저장
    print(f"🔄 임베딩 생성 및 Chroma DB 저장 중...")
    embeddings = _get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=child_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print(f"✅ Vector DB 구축 완료! ({CHROMA_DIR})")
    print(f"   저장된 자식 청크 수: {len(child_docs)}")
    return vectorstore


def get_vectorstore():
    """기존에 구축된 Vector DB를 로드합니다."""
    if not os.path.exists(CHROMA_DIR):
        return None

    embeddings = _get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    # 실제 데이터가 있는지 확인
    try:
        count = vectorstore._collection.count()
        if count == 0:
            return None
        return vectorstore
    except Exception:
        return None


def search_products(query: str, k: int = 3) -> List[Document]:
    """
    하이브리드 검색(Dense Vector + BM25 Sparse) 및 Cross-Encoder 리랭킹을 통해
    관련이 가장 높은 최적의 부모 문서(Parent Document)들을 반환합니다.
    """
    vectorstore = get_vectorstore()
    if vectorstore is None:
        print("⚠️ Vector DB를 찾을 수 없습니다. 빈 결과를 반환합니다.")
        return []

    # 1. ChromaDB에 저장된 모든 자식 문서 로드 (BM25 색인용)
    all_data = vectorstore.get()
    if not all_data or not all_data["documents"]:
        return []

    # get() 메서드로 반환된 데이터를 Document 객체 리스트로 복원
    all_docs = []
    for doc_id, text, metadata in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
        all_docs.append(Document(page_content=text, metadata=metadata))

    # 2. 1차 후보군 추출 개수 설정 (최대 후보 15개)
    candidates_limit = max(k * 4, 15)

    # 3. Dense 검색 (유사도 검색)
    dense_results = vectorstore.similarity_search(query, k=candidates_limit)

    # 4. Sparse 검색 (BM25 검색)
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = candidates_limit
    sparse_results = bm25_retriever.invoke(query)

    # 5. 하이브리드 결과 병합 및 자식 청크 수준 중복 제거
    seen_ids = set()
    combined_candidates = []
    
    # 두 결과를 교대로 섞어서 순위 가중치 보완 (RRF 느낌의 교차 병합)
    for d_doc, s_doc in zip(dense_results + [None] * len(sparse_results), sparse_results + [None] * len(dense_results)):
        if d_doc and d_doc.page_content not in seen_ids:
            seen_ids.add(d_doc.page_content)
            combined_candidates.append(d_doc)
        if s_doc and s_doc.page_content not in seen_ids:
            seen_ids.add(s_doc.page_content)
            combined_candidates.append(s_doc)

    combined_candidates = [c for c in combined_candidates if c is not None]
    if not combined_candidates:
        return []

    # 6. Cross-Encoder 리랭킹 (Reranker)
    reranker = _get_reranker()
    
    # 쿼리와 각 후보 문서의 '부모 본문(parent_content)'을 쌍으로 구성하여 예측
    # 자식 청크 텍스트보다는 전체 흐름이 들어있는 부모 정보로 연관성을 비교하는 것이 핵심입니다.
    pairs = []
    for doc in combined_candidates:
        parent_txt = doc.metadata.get("parent_content", doc.page_content)
        pairs.append([query, parent_txt])

    # 리랭커 점수 계산
    scores = reranker.predict(pairs)
    
    # 문서와 점수를 매핑하여 정렬
    ranked_docs = sorted(
        zip(combined_candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # 7. 최종 결과 취합 및 부모 문서 수준 중복 제거 (Deduplicate Parent Chunks)
    final_docs = []
    seen_parents = set()

    for doc, score in ranked_docs:
        parent_txt = doc.metadata.get("parent_content", doc.page_content)
        parent_id = doc.metadata.get("parent_id", parent_txt)
        
        if parent_id not in seen_parents:
            seen_parents.add(parent_id)
            # LLM에 전달할 문서를 부모 본문으로 승격(Promote)하여 생성
            parent_doc = Document(
                page_content=parent_txt,
                metadata={
                    "source_file": doc.metadata.get("source_file", "알 수 없음"),
                    "page": doc.metadata.get("page", "?"),
                    "score": float(score)  # 디버깅 및 분석용 점수 주입
                }
            )
            final_docs.append(parent_doc)
            
        if len(final_docs) >= k:
            break

    return final_docs
