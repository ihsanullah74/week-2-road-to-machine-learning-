import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ──────────────────────────────────────────
st.set_page_config(page_title="Food Habits Dashboard", layout="wide")
st.title("🍔 Student Food Habits Dashboard")
st.markdown("An interactive analysis of student eating habits, GPA, and lifestyle.")

# ── Load & Clean Data ─────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("food_coded.csv")
    cols = ['GPA','Gender','breakfast','calories_chicken',
            'calories_day','coffee','cook','eating_out',
            'exercise','fav_food','fruit_day','veggies_day',
            'vitamins','weight','healthy_feeling','income',
            'self_perception_weight']
    df = df[cols]
    df['GPA']    = pd.to_numeric(df['GPA'],    errors='coerce')
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    df = df.dropna(subset=['GPA','weight'])
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df['Gender_label'] = df['Gender'].map({1:'Female', 2:'Male'})
    return df

df = load_data()

# ── Sidebar Filters ───────────────────────────────────────
st.sidebar.header("🔍 Filters")
gender_opt = st.sidebar.multiselect("Gender",
    options=df['Gender_label'].unique(),
    default=df['Gender_label'].unique())

gpa_range = st.sidebar.slider("GPA Range",
    float(df['GPA'].min()), float(df['GPA'].max()),
    (float(df['GPA'].min()), float(df['GPA'].max())))

filtered = df[
    (df['Gender_label'].isin(gender_opt)) &
    (df['GPA'] >= gpa_range[0]) &
    (df['GPA'] <= gpa_range[1])
]

# ── KPI Cards ─────────────────────────────────────────────
st.subheader("📊 Key Statistics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Students",   len(filtered))
c2.metric("Average GPA",      round(filtered['GPA'].mean(), 2))
c3.metric("Average Weight",   round(filtered['weight'].mean(), 1))
c4.metric("Avg Healthy Feel", round(filtered['healthy_feeling'].mean(), 1))

st.markdown("---")

# ── Dataset Preview ───────────────────────────────────────
st.subheader("📋 Dataset Preview")
st.dataframe(filtered.head(10), use_container_width=True)

st.markdown("---")

# ── Charts Row 1 ──────────────────────────────────────────
st.subheader("📈 Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**GPA Distribution**")
    fig, ax = plt.subplots(figsize=(5,3))
    sns.histplot(filtered['GPA'], bins=12, color='steelblue', kde=True, ax=ax)
    ax.set_xlabel("GPA"); ax.set_ylabel("Count")
    st.pyplot(fig)

with col2:
    st.markdown("**Gender Distribution**")
    fig, ax = plt.subplots(figsize=(5,3))
    sns.countplot(x='Gender_label', data=filtered,
                  palette='viridis', hue='Gender_label',
                  legend=False, ax=ax)
    ax.set_xlabel("Gender"); ax.set_ylabel("Count")
    st.pyplot(fig)

# ── Charts Row 2 ──────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.markdown("**Exercise vs GPA**")
    fig, ax = plt.subplots(figsize=(5,3))
    sns.boxplot(x='exercise', y='GPA', data=filtered,
                palette='Set2', hue='exercise',
                legend=False, ax=ax)
    ax.set_xlabel("Exercise (1=Daily … 5=Never)")
    ax.set_ylabel("GPA")
    st.pyplot(fig)

with col4:
    st.markdown("**Calories in Chicken Meal**")
    fig, ax = plt.subplots(figsize=(5,3))
    bins   = [0, 400, 700, 1000, 2000]
    labels = ['Low','Medium','High','Very High']
    cal = pd.cut(filtered['calories_chicken'], bins=bins, labels=labels)
    cal.value_counts().plot.pie(autopct='%1.1f%%',
        colors=sns.color_palette('pastel'), ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

# ── Heatmap ───────────────────────────────────────────────
st.markdown("**Correlation Heatmap**")
fig, ax = plt.subplots(figsize=(9,5))
cols = ['GPA','weight','exercise','fruit_day',
        'veggies_day','healthy_feeling','eating_out']
sns.heatmap(filtered[cols].corr(), annot=True,
            cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)

st.markdown("---")
st.caption("Week 2 – Data Analysis Project | Ihsanullah Tanoli")
