import streamlit as st
import json
import pandas as pd
from pathlib import Path
from src.recommender import Recommender

st.set_page_config(page_title="AI Teaching Assistant", layout="wide", page_icon="🎓")

st.title("🎓 AI Teaching Assistant")
st.caption("Адаптивная система рекомендаций учебных вопросов с объяснимостью")

@st.cache_resource
def load_recommender():
    model_dir = Path(__file__).parent / "saved_model"
    return Recommender(str(model_dir))

try:
    recommender = load_recommender()
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()

if 'trajectory' not in st.session_state:
    st.session_state.trajectory = []
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []

col_left, col_right = st.columns([1, 2])

with col_left:
    st.header("📍 Траектория студента")
    
    new_question = st.text_input(
        "Добавить вопрос:",
        placeholder="Введите ID (например: q1234)",
        key="question_input"
    )
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("➕ Добавить", use_container_width=True):
            if new_question and new_question in recommender.item2idx:
                if new_question not in st.session_state.trajectory:
                    st.session_state.trajectory.append(new_question)
                    st.success(f"✅ Добавлен {new_question}")
                else:
                    st.warning("️ Уже в траектории")
            elif new_question:
                st.error(f"❌ Вопрос {new_question} не найден")
            st.rerun()
    
    with col_btn2:
        if st.button("🗑️ Очистить", use_container_width=True):
            st.session_state.trajectory = []
            st.session_state.recommendations = []
            st.rerun()
    
    if st.session_state.trajectory:
        st.markdown("---")
        st.markdown(f"**В траектории: {len(st.session_state.trajectory)} вопросов**")
        
        for i, question in enumerate(st.session_state.trajectory, 1):
            cluster_id = recommender.item_to_cluster.get(question, -1)
            col_q, col_del = st.columns([4, 1])
            with col_q:
                st.markdown(f"{i}. `{question}` (Тема {cluster_id})")
            with col_del:
                if st.button("❌", key=f"del_{question}"):
                    st.session_state.trajectory.remove(question)
                    st.session_state.recommendations = []
                    st.rerun()
        
        st.markdown("---")
        if st.button(" Получить рекомендации", type="primary", use_container_width=True):
            with st.spinner("Анализирую траекторию..."):
                recommendations = recommender.get_recommendations(
                    st.session_state.trajectory, 
                    top_k=5
                )
                st.session_state.recommendations = recommendations
                st.rerun()
    else:
        st.info("💡 Добавьте вопросы в траекторию")

with col_right:
    st.header("📚 Каталог вопросов")
    
    display_mode = st.radio(
        "Режим отображения:",
        ["📋 По ID", "🗂️ По темам"],
        horizontal=True
    )
    
    search_query = st.text_input(
        "🔍 Поиск по вопросам:",
        placeholder="Введите часть ID (например: q123)",
        key="search"
    )
    
    st.markdown("---")
    
    if display_mode == "📋 По ID":
        all_items = recommender.get_all_items()
        
        if search_query:
            filtered_items = [item for item in all_items if search_query.lower() in item.lower()]
        else:
            filtered_items = all_items
        
        st.caption(f"Всего вопросов: {len(filtered_items)} из {len(all_items)}")
        
        items_per_page = 50
        total_pages = (len(filtered_items) + items_per_page - 1) // items_per_page
        
        if total_pages > 1:
            page = st.slider("Страница:", 1, total_pages, 1)
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            page_items = filtered_items[start_idx:end_idx]
        else:
            page_items = filtered_items
        
        for item in page_items:
            cluster_id = recommender.item_to_cluster.get(item, -1)
            if st.button(f"{item} (Тема {cluster_id})", key=f"item_{item}", use_container_width=True):
                if item not in st.session_state.trajectory:
                    st.session_state.trajectory.append(item)
                    st.success(f"✅ Добавлен {item}")
                    st.rerun()
                else:
                    st.warning("⚠️ Уже в траектории")
    
    else:
        st.caption(f"Вопросы сгруппированы по {len(recommender.question_clusters)} темам")
        
        if search_query:
            all_items = recommender.get_all_items()
            filtered_items = [item for item in all_items if search_query.lower() in item.lower()]
            filtered_clusters = {}
            for item in filtered_items:
                cluster_id = recommender.item_to_cluster.get(item, -1)
                if cluster_id not in filtered_clusters:
                    filtered_clusters[cluster_id] = []
                filtered_clusters[cluster_id].append(item)
        else:
            filtered_clusters = recommender.question_clusters
        
        for cluster_id in sorted(filtered_clusters.keys()):
            items = filtered_clusters[cluster_id]
            with st.expander(f"📁 Тема {cluster_id} ({len(items)} вопросов)", expanded=False):
                for item in items:
                    if st.button(item, key=f"cluster_{cluster_id}_{item}", use_container_width=True):
                        if item not in st.session_state.trajectory:
                            st.session_state.trajectory.append(item)
                            st.success(f"✅ Добавлен {item}")
                            st.rerun()
                        else:
                            st.warning("⚠️ Уже в траектории")

if st.session_state.recommendations:
    st.markdown("---")
    st.header("🎯 Рекомендуемые следующие шаги")
    
    for i, rec in enumerate(st.session_state.recommendations, 1):
        with st.container():
            st.markdown(f"**{i}. Вопрос: `{rec['item_name']}` (Тема {rec['cluster_id']})**")
            
            explanation_key = f"explain_{i}_{rec['item_name']}"
            if st.button(" Почему этот вопрос?", key=explanation_key):
                with st.spinner("Генерирую объяснение..."):
                    explanation_data = recommender.get_explanation(
                        st.session_state.trajectory,
                        rec['item_name'],
                        top_k_influence=2
                    )
                
                if explanation_data:
                    st.info(f"**Объяснение:** {explanation_data['explanation']}")
                    
                    st.markdown("**Сходство с вопросами из истории:**")
                    
                    influence_data = {
                        q['question']: round(q['similarity'] * 100, 1) 
                        for q in explanation_data['influential_questions']
                    }
                    
                    df_influence = pd.DataFrame([influence_data]).T
                    df_influence.columns = ['Сходство (%)']
                    st.bar_chart(df_influence)
                    
                    st.caption(f"Тема рекомендуемого вопроса: {explanation_data['recommended_cluster']}")

with st.sidebar:
    st.header("ℹ️ О модели")
    st.markdown(f"""
    - **Архитектура**: {recommender.hyperparams['model_type']}
    - **Вопросов в базе**: {len(recommender.get_all_items())}
    - **Тем **: {len(recommender.question_clusters)}
    """)