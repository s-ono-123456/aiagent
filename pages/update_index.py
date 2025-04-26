import streamlit as st
import os
import glob
from document_store import DocumentStore

# タイトルとページ説明
st.title("インデックス更新")
st.write("ドキュメントのインデックスを更新するページです。")

# ドキュメントストアの初期化
@st.cache_resource
def get_document_store():
    doc_store = DocumentStore()
    return doc_store

# ドキュメントストアを取得
doc_store = get_document_store()

# 現在のインデックス情報を表示
st.subheader("現在のインデックス情報")
categories = doc_store.get_all_categories()

if not categories:
    st.info("現在、インデックスは作成されていません。")
else:
    st.write(f"インデックス数: {len(categories)}")
    st.write("カテゴリ一覧:")
    for category in categories:
        st.write(f"- {category}")

# インデックス更新セクション
st.subheader("インデックス更新")

# サンプルディレクトリのパスを取得
sample_dir = "sample"

if not os.path.exists(sample_dir):
    st.error(f"ディレクトリ '{sample_dir}' が存在しません。")
else:
    # サブディレクトリを取得
    subdirs = [d for d in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, d))]
    
    if not subdirs:
        st.warning(f"'{sample_dir}' ディレクトリ内にサブディレクトリが見つかりません。")
    else:
        st.write(f"'{sample_dir}' ディレクトリ内のサブディレクトリ:")
        for subdir in subdirs:
            md_files = glob.glob(os.path.join(sample_dir, subdir, "*.md"))
            st.write(f"- {subdir} ({len(md_files)}ファイル)")
        
        # 更新対象を選択
        update_options = ["すべてのカテゴリ"] + subdirs
        update_selection = st.selectbox("更新対象", update_options)
        
        if st.button("インデックスを更新"):
            with st.spinner("インデックスを更新中..."):
                if update_selection == "すべてのカテゴリ":
                    # すべてのカテゴリを更新
                    doc_store.load_documents_from_directory(sample_dir)
                    st.success("すべてのカテゴリのインデックスを更新しました。")
                else:
                    # 選択したカテゴリのみ更新
                    category = update_selection
                    md_files = glob.glob(os.path.join(sample_dir, category, "*.md"))
                    
                    texts = []
                    metadatas = []
                    
                    for md_file in md_files:
                        try:
                            with open(md_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                                
                            # ファイル情報をメタデータとして保存
                            metadata = {
                                "source": md_file,
                                "filename": os.path.basename(md_file),
                                "category": category
                            }
                            
                            texts.append(content)
                            metadatas.append(metadata)
                        except Exception as e:
                            st.error(f"ファイル読み込みエラー ({md_file}): {e}")
                    
                    if texts:
                        # 既存のインデックスを削除（更新のため）
                        if category in doc_store.vectorstores:
                            del doc_store.vectorstores[category]
                        if category in doc_store.documents:
                            doc_store.documents[category] = []
                        
                        # 新しいドキュメントを追加
                        doc_store.add_documents(texts, metadatas, category)
                        doc_store.save_index(category)
                        st.success(f"カテゴリ '{category}' のインデックスを更新しました。")
                    else:
                        st.warning(f"カテゴリ '{category}' に読み込めるファイルがありませんでした。")

# インデックスの削除セクション
st.subheader("インデックス削除")

# 削除対象を選択
if categories:
    delete_options = ["選択してください"] + categories
    delete_selection = st.selectbox("削除対象", delete_options)
    
    if delete_selection != "選択してください" and st.button("インデックスを削除"):
        category = delete_selection
        
        # インデックスファイルを削除
        index_file = os.path.join(doc_store.index_dir, f"{category}.faiss")
        index_file_pkl = os.path.join(doc_store.index_dir, f"{category}.pkl")
        
        try:
            if os.path.exists(index_file):
                os.remove(index_file)
            if os.path.exists(index_file_pkl):
                os.remove(index_file_pkl)
                
            # メモリ上のインデックスも削除
            if category in doc_store.vectorstores:
                del doc_store.vectorstores[category]
            if category in doc_store.documents:
                doc_store.documents[category] = []
                
            st.success(f"カテゴリ '{category}' のインデックスを削除しました。")
            
            # カテゴリ一覧を更新
            categories = doc_store.get_all_categories()
            
        except Exception as e:
            st.error(f"インデックス削除エラー: {e}")
else:
    st.info("削除可能なインデックスはありません。")