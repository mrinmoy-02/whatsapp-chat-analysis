import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stButton>button {
        color: white;
        background-color: #ff4b4b;
        border-radius: 10px;
        height: 40px;
        width: 100%;
        font-size: 16px;
    }
    .stFileUploader>label {
        color: #ff4b4b;
        font-size: 16px;
        font-weight: bold;
    }
    .stSelectbox>label {
        color: #ff4b4b;
        font-size: 16px;
        font-weight: bold;
    }
    .st-c3 {
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True
)

# App title
st.sidebar.title("📊 Whatsapp Chat Analyzer")

# File uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("📂 Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # Fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):
        st.sidebar.markdown("<hr>", unsafe_allow_html=True)
        
        # Stats Area
        st.title("📈 Top Statistics")
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.markdown(
            """
            <style>
            .stat-card {
                border-radius: 15px;
                padding: 20px;
                margin: 10px;
                background-color: #f8f9fa;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                text-align: center;
            }
            .stat-title {
                font-size: 18px;
                color: #333;
            }
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                color: #ff4b4b;
            }
            </style>
            """, unsafe_allow_html=True
        )
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="stat-card"><div class="stat-title">Total Messages</div><div class="stat-value">{}</div></div>'.format(num_messages), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="stat-card"><div class="stat-title">Total Words</div><div class="stat-value">{}</div></div>'.format(words), unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="stat-card"><div class="stat-title">Media Shared</div><div class="stat-value">{}</div></div>'.format(num_media_messages), unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="stat-card"><div class="stat-title">Links Shared</div><div class="stat-value">{}</div></div>'.format(num_links), unsafe_allow_html=True)

        # Monthly timeline
        st.title("📅 Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily timeline
        st.title("📆 Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity map
        st.title("🗺️ Activity Map")
        col1, col2 = st.columns(2)
        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("📊 Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap, cmap="YlGnBu")
        st.pyplot(fig)

        # Finding the busiest users in the group (Group level)
        if selected_user == 'Overall':
            st.title("👥 Most Busy Users")
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)
            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("☁️ Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        # Most common words
        st.title("📚 Most Common Words")
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1], color='blue')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Emoji analysis
        st.title("😊 Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f%%", colors=sns.color_palette("coolwarm"))
            st.pyplot(fig)

        # Sentiment Analysis
        st.title("🔍 Sentiment Analysis")
        sentiments = helper.sentiment_analysis(selected_user, df)
        st.write("Average Sentiment Polarity: ", sentiments.mean())
        fig, ax = plt.subplots()
        sns.histplot(sentiments, bins=20, kde=True, ax=ax, color='blue')
        st.pyplot(fig)

        # Message Length Analysis
        st.title("✉️ Message Length Analysis")
        msg_length = helper.message_length_analysis(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(msg_length['date'], msg_length['message_length'], color='red')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Response Time Analysis
        st.title("⏳ Response Time Analysis")
        response_time = helper.response_time_analysis(selected_user, df)
        st.write("Average Response Time (seconds): ", response_time['response_time_seconds'].mean())
        fig, ax = plt.subplots()
        sns.histplot(response_time['response_time_seconds'], bins=20, kde=True, ax=ax, color='green')
        st.pyplot(fig)

        # Active Hours Analysis
        st.title("🕒 Active Hours Analysis")
        active_hours = helper.active_hours_analysis(selected_user, df)
        fig, ax = plt.subplots()
        ax.bar(active_hours.index, active_hours.values, color='purple')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Media Types Shared Analysis
        st.title("📷 Media Types Shared Analysis")
        media_types_count = helper.media_types_shared_analysis(selected_user, df)
        fig, ax = plt.subplots()
        ax.pie(media_types_count.values, labels=media_types_count.index, autopct="%0.2f%%", colors=sns.color_palette("coolwarm"))
        st.pyplot(fig)
