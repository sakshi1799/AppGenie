#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
import io
import re
#from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import streamlit as st
import re
import numpy as np


# In[ ]:


#langchain agent create_pandas_dataframe_agent is for LLM to interact with a Pandas DataFrame
import streamlit as st

OPENAI_API_KEY = "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"

def read_from_s3(file_name):
    s3 = boto3.client("s3",aws_access_key_id="",aws_secret_access_key="")
    obj = s3.get_object(Bucket='capstonebucket11', Key= file_name)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    return df

app_infos_df = read_from_s3('app_infos_df.csv')
fitness_app_infos_df = read_from_s3('fitness_app_infos_df.csv')
health_app_infos_df = read_from_s3('health_app_infos_df.csv')


reviews_df = read_from_s3('app_reviews_df.csv')
fitness_app_reviews_df = read_from_s3('fitness_app_reviews_df.csv')
health_app_reviews_df = read_from_s3('health_app_reviews_df.csv')

def app_questions(df_name):
    st.title('Questions Related to Apps 🙋')
    option = st.selectbox('Select an option', ('Ask a question about apps',))

    if option == 'Ask a question about apps':
        question = st.text_input('Enter your question here:')
        if st.button('Get Answer'):
            chat = ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-4", temperature=0.0)
            agent = create_pandas_dataframe_agent(chat, df_name, verbose=True)
            response = agent.run(question)
            st.write(f"Answer: {response}")

def review_summarization(df_name):
    st.title('Reviews Summarization 📖')
    option = st.selectbox('Select an option', ('Get reviews summary',))

    if option == 'Get reviews summary':
        question = st.text_input('Enter your review summary query here:')
        if st.button('Get Summary'):
            chat = ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-4", temperature=0.4)
            agent_reviews = create_pandas_dataframe_agent(chat, df_name, verbose=False)
            response = agent_reviews.run(question)
            st.write(f"Summary: {response}")

import re
import numpy as np
# Function to calculate mean price from a price range string
def calculate_mean_price(price_range):
    if price_range is None or price_range.lower() == 'none':
        return 0

    # Extract numeric values using regular expressions
    prices = re.findall(r'\d+\.\d+', price_range)

    if len(prices) > 0:
        # Convert extracted prices to float and calculate mean
        prices = [float(price) for price in prices]
        return np.mean(prices)
    else:
        return 0

def convert(df):
    df['inAppProductPrice'] = df['inAppProductPrice'].astype(str).apply(lambda x: re.sub(r'[^a-zA-Z0-9\s.]', '', x))

    # Calculate mean prices for 'inAppProductPrice' column
    mean_prices = df['inAppProductPrice'].apply(calculate_mean_price)

    # Create a new DataFrame with mean values
    mean_prices_df = pd.DataFrame({'Mean_InAppProductPrice': mean_prices})

    mean_prices_df['App_Name'] = app_infos_df['title']
    return mean_prices_df

mean_productivity = convert(app_infos_df)
mean_fitness = convert(fitness_app_infos_df)
mean_health = convert(health_app_infos_df)

def app_visualizations(df_name, mean_df):
    st.title('Visualizations 📊')
    option = st.selectbox('Select an option', ('Get top 5 apps by ratings',
                                              'Get bottom 3 apps by ratings',
                                              'Get 5 apps with the most number of installs',
                                              'Visualize mean in-app prices of all apps'))

    if option == 'Get top 5 apps by ratings':
        top_apps = df_name.sort_values(by='ratings', ascending=False).head(5)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='ratings', y='title', data=top_apps, ax=ax)
        ax.set_xlabel('Ratings')
        ax.set_ylabel('Apps')
        ax.set_title('Top 5 Apps by Ratings')
        st.pyplot(fig)

    elif option == 'Get bottom 3 apps by ratings':
        bottom_apps = df_name.sort_values(by='ratings').head(3)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='ratings', y='title', data=bottom_apps, ax=ax, color='red')
        ax.set_xlabel('Ratings')
        ax.set_ylabel('Apps')
        ax.set_title('Bottom 3 Apps by Ratings')
        st.pyplot(fig)

    elif option == 'Get 5 apps with the most number of installs':
        top_installs = df_name.sort_values(by='realInstalls', ascending=False).head(5)
        
        fig, ax = plt.subplots()
        ax.pie(top_installs['realInstalls'], labels=top_installs['title'], autopct='%1.1f%%')
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.set_title('Pie Chart for Top 5 Apps by Installs')

        st.pyplot(fig)

    elif option == 'Visualize mean in-app prices of all apps':
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x= 'App_Name', y = 'Mean_InAppProductPrice', data = mean_df, ax=ax)
        ax.set_xticklabels(mean_df['App_Name'], rotation=45)
        ax.set_xlabel('Apps')
        ax.set_ylabel('Mean In-App Prices')
        ax.set_title('Mean In-App Prices of Apps')
        st.pyplot(fig)

st.title('WellnessWizard-Your-Health-Fitness-Productivity-Guru 🎯')

def select_category():
    return st.sidebar.selectbox('Select an app category', ('Productivity 📝', 'Fitness 🏋️', 'Health 🏥'))

selected_category = select_category()

if selected_category == 'Productivity 📝':
    app_questions(app_infos_df)
    review_summarization(reviews_df)
    app_visualizations(app_infos_df, mean_productivity)

elif selected_category == 'Fitness 🏋️':
    app_questions(fitness_app_infos_df)
    review_summarization(fitness_app_reviews_df)
    app_visualizations(fitness_app_infos_df, mean_fitness)

elif selected_category == 'Health 🏥':
    app_questions(health_app_infos_df)
    review_summarization(health_app_reviews_df)
    app_visualizations(health_app_infos_df, mean_health)

