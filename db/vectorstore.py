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
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# ─── 경로 설정 ───
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

# ─── 임베딩 및 리랭커 모델 설정 ───
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
RERANKER_MODEL_NAME = "Dongjin-kr/ko-reranker"

# 프로세스 단위 싱글톤 캐시
_EMBEDDINGS_INSTANCE = None
_VECTORSTORE_INSTANCE = None
_ALL_DOCS_CACHE = None
_BM25_RETRIEVER_INSTANCE = None
_RERANKER_INSTANCE = None


def _get_embeddings():
    """HuggingFace 임베딩 모델 인스턴스를 반환합니다."""
    global _EMBEDDINGS_INSTANCE
    if _EMBEDDINGS_INSTANCE is None:
        _EMBEDDINGS_INSTANCE = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDINGS_INSTANCE


def _get_reranker() -> CrossEncoder:
    """리랭커 싱글톤 인스턴스를 반환합니다."""
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

    # 1단계: 부모 청크 분할 (상세 컨텍스트 유지용 - 1200자, 정보 유실 방지)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
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

    _reset_vectorstore_caches()

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
    _prime_vectorstore_caches(vectorstore, child_docs)
    return vectorstore


def get_vectorstore():
    """기존에 구축된 Vector DB를 로드합니다."""
    global _VECTORSTORE_INSTANCE

    if _VECTORSTORE_INSTANCE is not None:
        return _VECTORSTORE_INSTANCE

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
        _VECTORSTORE_INSTANCE = vectorstore
        return vectorstore
    except Exception:
        return None


def search_products(query: str, k: int = 3) -> List[Document]:
    """
    초경량 임베딩 모델을 사용한 유사도 검색(Dense)과 BM25 키워드 검색(Sparse)을 결합한
    하이브리드 검색 후보군을 추출한 후, Cross-Encoder 리랭킹을 거쳐 최종 k개의 부모 문서를 반환합니다.
    """
    vectorstore = get_vectorstore()
    if vectorstore is None:
        print("⚠️ Vector DB를 찾을 수 없습니다. 빈 결과를 반환합니다.")
        return []

    # 1. 각각의 검색 방식에서 추출할 1차 후보 수 (기본 k=4일 때 각각 15개 내외)
    candidates_limit = max(k * 4, 15)

    # A. Dense 검색 (유사도 검색)
    dense_results = vectorstore.similarity_search(query, k=candidates_limit)

    # B. Sparse (BM25) 검색 (키워드 매칭)
    all_docs = _get_all_docs(vectorstore)
    sparse_results = []
    if all_docs:
        sparse_results = _search_sparse_with_bm25(all_docs, query, candidates_limit)

    # C. 두 결과의 하이브리드 병합 및 자식 청크 중복 제거 (리랭커 부하 제어)
    seen_chunk_contents = set()
    merged_candidates = []

    # Dense 결과 먼저 삽입
    for doc in dense_results:
        chunk_txt = doc.page_content
        if chunk_txt not in seen_chunk_contents:
            seen_chunk_contents.add(chunk_txt)
            merged_candidates.append(doc)

    # Sparse 결과 추가 삽입
    for doc in sparse_results:
        chunk_txt = doc.page_content
        if chunk_txt not in seen_chunk_contents:
            seen_chunk_contents.add(chunk_txt)
            merged_candidates.append(doc)

    # 리랭킹할 후보군의 상한선 설정 (최대 30개)
    final_candidates = merged_candidates[:30]

    if not final_candidates:
        return []

    # 2. Cross-Encoder 리랭킹 (Reranker)
    reranker = _get_reranker()
    
    # 쿼리와 각 후보 문서의 '부모 본문(parent_content)'을 쌍으로 구성하여 예측
    pairs = []
    for doc in final_candidates:
        parent_txt = doc.metadata.get("parent_content", doc.page_content)
        pairs.append([query, parent_txt])

    # 리랭커 점수 계산
    scores = reranker.predict(pairs)
    
    # 3. 단일 상품 집중 질문의 경우 동일 파일(source_file) 부스팅 가산점 적용
    # 비교 질문이 아닐 때, 1위로 매치된 핵심 상품의 다른 페이지들이 다른 상품에 밀려 누락되는 것 방지
    is_comparative = any(kw in query for kw in ["비교", "차이", "모두", "목록", "공통", "다른점", "차이점"])
    target_file = None
    if not is_comparative and dense_results:
        target_file = dense_results[0].metadata.get("source_file")

    boosted_candidates = []
    for doc, score in zip(final_candidates, scores):
        doc_source = doc.metadata.get("source_file")
        final_score = float(score)
        if target_file and doc_source == target_file:
            final_score += 0.5  # 동일 파일의 다른 페이지 조각들에 부스팅 가산점 부여
        boosted_candidates.append((doc, final_score))

    # 문서와 점수를 매핑하여 정렬
    ranked_docs = sorted(
        boosted_candidates,
        key=lambda x: x[1],
        reverse=True
    )

    # 4. 최종 결과 취합 및 부모 문서 수준 중복 제거 (Deduplicate Parent Chunks)
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


def _search_sparse_with_bm25(
    all_docs: List[Document],
    query: str,
    candidates_limit: int,
) -> List[Document]:
    global _BM25_RETRIEVER_INSTANCE

    try:
        from langchain_community.retrievers import BM25Retriever
    except Exception as e:
        print(f"Warning: BM25 retriever import failed, using dense search only. error={e}")
        return []

    try:
        if _BM25_RETRIEVER_INSTANCE is None:
            _BM25_RETRIEVER_INSTANCE = BM25Retriever.from_documents(all_docs)

        _BM25_RETRIEVER_INSTANCE.k = candidates_limit
        return _BM25_RETRIEVER_INSTANCE.invoke(query)
    except Exception as e:
        print(f"Warning: BM25 retriever failed, using dense search only. error={e}")
        return []


def _get_all_docs(vectorstore) -> List[Document]:
    global _ALL_DOCS_CACHE

    if _ALL_DOCS_CACHE is not None:
        return _ALL_DOCS_CACHE

    all_data = vectorstore.get()
    if not all_data or not all_data.get("documents"):
        return []

    docs = []
    for text, metadata in zip(all_data["documents"], all_data["metadatas"]):
        docs.append(Document(page_content=text, metadata=metadata))

    _ALL_DOCS_CACHE = docs
    return docs


def _prime_vectorstore_caches(vectorstore, child_docs: List[Document]) -> None:
    global _VECTORSTORE_INSTANCE, _ALL_DOCS_CACHE, _BM25_RETRIEVER_INSTANCE

    _VECTORSTORE_INSTANCE = vectorstore
    _ALL_DOCS_CACHE = list(child_docs)
    _BM25_RETRIEVER_INSTANCE = None


def _reset_vectorstore_caches() -> None:
    global _VECTORSTORE_INSTANCE, _ALL_DOCS_CACHE, _BM25_RETRIEVER_INSTANCE

    _VECTORSTORE_INSTANCE = None
    _ALL_DOCS_CACHE = None
    _BM25_RETRIEVER_INSTANCE = None
