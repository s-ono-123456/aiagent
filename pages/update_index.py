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

# カテゴリ情報を取得（各タブで使用）
categories = doc_store.get_all_categories()

# タブの作成
tab_info, tab_update, tab_add, tab_delete = st.tabs([
    "インデックス情報", "インデックス更新", "ドキュメント追加", "インデックス削除"
])

# タブ1: インデックス情報
with tab_info:
    st.subheader("現在のインデックス情報")
    
    if not categories:
        st.info("現在、インデックスは作成されていません。")
    else:
        st.write(f"インデックス数: {len(categories)}")
        st.write("カテゴリ一覧:")
        for category in categories:
            st.write(f"- {category}")

# タブ2: インデックス更新
with tab_update:
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

# タブ3: ドキュメント追加
with tab_add:
    st.subheader("ドキュメント追加")
    
    # サンプルディレクトリの確認
    sample_dir = "sample"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
        st.info(f"'{sample_dir}' ディレクトリを作成しました。")
    
    # カテゴリの選択（既存または新規）
    if not categories:
        category_options = ["新規カテゴリ"]
    else:
        category_options = categories + ["新規カテゴリ"]
    
    selected_category = st.selectbox("カテゴリ", category_options, key="add_category_select")
    
    # 新規カテゴリの場合、カテゴリ名の入力を表示
    if selected_category == "新規カテゴリ":
        new_category = st.text_input("新規カテゴリ名", "")
        current_category = new_category
    else:
        current_category = selected_category
    
    # カテゴリが選択または入力されている場合のみドキュメント追加UIを表示
    if current_category:
        # テキスト入力エリア
        doc_content = st.text_area("ドキュメント内容", height=200)
        
        # ファイル名の入力（オプション）
        # 新規カテゴリまたはカテゴリにドキュメントがない場合のエラー回避
        doc_count = 1
        try:
            doc_count = len(doc_store.get_documents(current_category)) + 1
        except:
            doc_count = 1
            
        doc_filename = st.text_input("ファイル名（オプション）", f"document_{doc_count}.md")
        
        # 追加ボタン
        if st.button("ドキュメントを追加"):
            if not doc_content:
                st.error("ドキュメント内容を入力してください。")
            else:
                # ファイル名が指定されていない場合はデフォルト名を使用
                if not doc_filename:
                    doc_filename = f"document_{len(doc_store.get_documents(current_category)) + 1}.md"
                
                # ファイル名が.mdで終わることを確認
                if not doc_filename.endswith(".md"):
                    doc_filename += ".md"
                
                # カテゴリディレクトリが存在しない場合は作成
                category_dir = os.path.join(sample_dir, current_category)
                if not os.path.exists(category_dir):
                    os.makedirs(category_dir)
                
                # ファイルの書き込み
                file_path = os.path.join(category_dir, doc_filename)
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(doc_content)
                    
                    # メタデータを作成
                    metadata = {
                        "source": file_path,
                        "filename": doc_filename,
                        "category": current_category
                    }
                    
                    # ドキュメントストアに追加
                    doc_store.add_documents([doc_content], [metadata], current_category)
                    doc_store.save_index(current_category)
                    
                    st.success(f"ドキュメント '{doc_filename}' をカテゴリ '{current_category}' に追加しました。")
                except Exception as e:
                    st.error(f"ドキュメント追加エラー: {e}")
    else:
        st.info("ドキュメントを追加するには、カテゴリを選択または入力してください。")

# タブ4: インデックス削除
with tab_delete:
    st.subheader("インデックス削除")
    
    # 削除対象を選択
    if categories:
        delete_options = ["選択してください"] + categories
        delete_selection = st.selectbox("削除対象", delete_options, key="delete_category_select")
        
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