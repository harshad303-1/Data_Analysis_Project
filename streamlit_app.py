import os 
import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv
import matplotlib.pyplot as plt
import numpy as np

# Cache the LLM initialization
@st.cache_resource
def get_llm():
    """Initialize and return the OpenAI LLM instance"""
    return OpenAI(temperature=0)

# Cache the pandas agent creation
@st.cache_resource
def get_pandas_agent(_llm, df):
    """Create and return the pandas dataframe agent"""
    return create_pandas_dataframe_agent(_llm, df, verbose=True,handle_parsing_errors=True, allow_dangerous_code=True)

def create_visualization(df, column_name, viz_type, title=None):
    """Create different types of visualizations using Matplotlib based on the specified type"""
    try:
        # Create figure and axis objects
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if viz_type == "bar":
            if df[column_name].dtype in ['object', 'category']:
                value_counts = df[column_name].value_counts()
                value_counts.plot(kind='bar', ax=ax)
            else:
                df[column_name].plot(kind='bar', ax=ax)
                
        elif viz_type == "line":
            # Handle both series and dataframe inputs
            if isinstance(df, pd.DataFrame):
                if pd.api.types.is_numeric_dtype(df[column_name]):
                    # Create line plot with index as x-axis
                    ax.plot(df.index, df[column_name])
                    ax.set_ylabel(column_name)
                else:
                    raise ValueError(f"Column {column_name} must be numeric for line plots")
            else:
                ax.plot(df.index, df)
            
        elif viz_type == "scatter":
            ax.scatter(df.index, df[column_name])
            
        elif viz_type == "histogram":
            # Ensure the data is numeric and handle NaN values
            if pd.api.types.is_numeric_dtype(df[column_name]):
                data = df[column_name].dropna()
                
                # Calculate optimal number of bins using Sturges' rule
                n_bins = int(np.log2(len(data)) + 1)
                
                # Create histogram with density=False to show counts
                ax.hist(data, bins=n_bins, edgecolor='black', alpha=0.7)
                
                # Add mean and median lines
                mean_val = data.mean()
                median_val = data.median()
                ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
                
                ax.set_ylabel('Frequency')
                ax.legend()
            else:
                raise ValueError(f"Column {column_name} must be numeric for histograms")
            
        elif viz_type == "box":
            df[column_name].plot(kind='box', ax=ax)
            
        elif viz_type == "pie":
            value_counts = df[column_name].value_counts()
            ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
            ax.axis('equal')
            
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")
        
        # Set title if provided
        if title:
            ax.set_title(title)
        
        # Customize the appearance
        ax.set_xlabel(column_name)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Display in Streamlit
        st.pyplot(fig)
        plt.close()
        return True
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return False

# Cache categorical data analysis
@st.cache_data(show_spinner="Analyzing categorical data...")
def analyze_categorical_data(_pandas_agent, column_name, df):
    st.write(f"Analysis of {column_name}")
    
    if column_name not in df.columns:
        st.error(f"Column '{column_name}' not found in the dataset")
        return
    
    try:
        if isinstance(df[column_name], pd.DataFrame):
            value_counts = df[column_name].iloc[:, 0].value_counts()
        else:
            value_counts = df[column_name].value_counts()
            
        st.write("Category Distribution:")
        st.write(value_counts)
        
        st.write("Distribution Visualization:")
        chart_data = pd.DataFrame({
            'Category': value_counts.index,
            'Count': value_counts.values
        })
        st.bar_chart(chart_data.set_index('Category'))
        
        distribution = _pandas_agent.run(f"""Analyze the distribution of {column_name} and provide insights. 
        Include:
        1. Most common categories
        2. Least common categories
        3. Any interesting patterns
        4. Potential business implications
        """)
        st.write("Key Insights:")
        st.write(distribution)
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        st.write("Raw data for debugging:")
        st.write(df[column_name].head())

# Cache question processing
# @st.cache_data(show_spinner="Processing question...", ttl="10m")
# def process_question(_pandas_agent, question, df):
#     """Process a data-related question and determine appropriate visualization"""
#     try:
#         visualization_keywords = ["show", "plot", "display", "visualize", "graph", "chart", "distribution"]
        
#         # Check if the question involves visualization
#         needs_viz = any(keyword in question.lower() for keyword in visualization_keywords)
        
#         if needs_viz:
#             # Ask LLM to analyze the question and data to determine visualization needs
#             viz_analysis = _pandas_agent.run(f"""
#             Analyze this question: "{question}"
#             For any columns that need visualization, provide a JSON-like response with:
#             1. The column name
#             2. The best visualization type (choose from: bar, line, scatter, histogram, box, pie)
#             3. A brief explanation of why this visualization type is appropriate
#             4. A title for the visualization
            
#             Consider:
#             - Data type of the column
#             - The question's intent
#             - Statistical properties of the data
            
#             Format: {{"column": "column_name", "viz_type": "type", "reason": "explanation", "title": "chart title"}}
#             If multiple visualizations are needed, provide multiple JSON objects.
#             """)
            
#             try:
#                 # Extract visualization recommendations from the LLM's response
#                 import re
#                 import json
                
#                 # Find all JSON-like objects in the response
#                 json_patterns = re.findall(r'\{[^}]+\}', viz_analysis)
                
#                 if json_patterns:
#                     for json_str in json_patterns:
#                         try:
#                             viz_info = json.loads(json_str)
#                             if all(k in viz_info for k in ["column", "viz_type", "reason", "title"]):
#                                 st.write(f"Visualization Insight: {viz_info['reason']}")
#                                 create_visualization(
#                                     df, 
#                                     viz_info['column'], 
#                                     viz_info['viz_type'],
#                                     viz_info['title']
#                                 )
#                         except json.JSONDecodeError:
#                             continue
#                 else:
#                     # Fallback to original visualization logic if no JSON patterns found
#                     columns_mentioned = [col for col in df.columns if col.lower() in question.lower()]
#                     if columns_mentioned:
#                         for col in columns_mentioned:
#                             if df[col].dtype in ['object', 'category'] or df[col].dtype == 'bool':
#                                 analyze_categorical_data(_pandas_agent, col, df)
#                             else:
#                                 st.line_chart(df, y=[col])
#                                 statistics = _pandas_agent.run(f"Analyze {col} and provide key statistical insights")
#                                 st.write(statistics)
            
#             except Exception as e:
#                 st.error(f"Error in visualization processing: {str(e)}")
        
#         # Get the answer from the agent
#         response = _pandas_agent.run(question)
#         return response
        
#     except Exception as e:
#         st.error(f"An error occurred: {str(e)}")
#         return f"I encountered an error while processing your question: {str(e)}"

def store_visualization(column, viz_type, title, stats=None):
    """Store visualization metadata in session state"""
    viz_data = {
        'column': column,
        'type': viz_type,
        'title': title,
        'stats': stats,
        'timestamp': pd.Timestamp.now()
    }
    st.session_state.visualization_history.append(viz_data)


@st.cache_data(show_spinner="Processing question...", ttl="10m")
def process_question(_pandas_agent, question, df):
    try:
        # Increase context window to last 5 conversations
        context = [f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history[-5:]]
        context_str = "\n".join(context)

        insights_keywords = ["insights", "analyze", "explain", "describe", "tell me about", "what do you see", "what can you tell"]
        is_analysis_request = any(keyword in question.lower() for keyword in insights_keywords)
       
        if is_analysis_request and st.session_state.visualization_history:
            last_viz = st.session_state.visualization_history[-1]
            # Check if it's a single or multi-column visualization
            columns_str = last_viz.get('column') if 'column' in last_viz else ', '.join(last_viz.get('columns', []))
            
            analysis_prompt = f"""
            Analyze this visualization of {columns_str} using actual data:
            1. For single column:
                - Distribution/frequency patterns
                - Key statistics (mean, median, mode)
                - Notable outliers
            2. For multiple columns:
                - Relationships/correlations
                - Trends across categories
                - Key differences between groups
            3. Business implications of the patterns
            Format: "Analysis: [your insights]"
            """
            response = _pandas_agent.run(analysis_prompt)
            if "Analysis:" in response:
                response = response.split("Analysis:")[1].strip()
            return response, None

        explanation_keywords = ["explain", "describe", "tell me about", "what does", "analyze", "interpret"]
        is_viz_explanation = any(keyword in question.lower() for keyword in explanation_keywords) and "graph" in question.lower()
       
        if is_viz_explanation and st.session_state.visualization_history:
           last_viz = st.session_state.visualization_history[-1]
           # Check if it's a single or multi-column visualization
           columns_str = last_viz.get('column') if 'column' in last_viz else ', '.join(last_viz.get('columns', []))
           analysis_prompt = f"""
           Analyze the {last_viz['type']} chart of {columns_str} that was just displayed.
           Provide insights about:
           1. The overall distribution/pattern
           2. Any notable trends or outliers
           3. Key statistics if relevant
           Keep the explanation clear and data-focused.
           """
           response = _pandas_agent.run(analysis_prompt)
           return response, None
        
        visualization_keywords = ["show", "plot", "display", "visualize", "graph", "chart", "distribution", "histogram", "box", "vs", "versus", "compare", "correlation", "visualization"]
        needs_viz = any(keyword in question.lower() for keyword in visualization_keywords)
        
        viz_data = None
        if needs_viz:
            viz_query = _pandas_agent.run(f"""
            For the question: "{question}"
            1. Identify which columns need to be visualized
            2. Choose the best visualization type (scatter/bar/line/box/pie)
            3. Return ONLY in this format:
            COLUMNS: column1, column2 (if comparing two columns)
            TYPE: <viz_type>
            
            Base your choice on:
            - If comparing categories vs numbers: use bar
            - If comparing two numbers: use scatter
            - If looking at trends: use line
            - If analyzing distributions with categories: use box
            - If showing value distributions: use histogram
            - If showing proportions of a whole: use pie  # Add this line
            
            Consider the columns' data types and question context carefully.
            """)
            
            try:
                lines = viz_query.strip().split('\n')
                columns = []
                viz_type = None
                
                for line in lines:
                    if line.startswith('COLUMNS:'):
                        columns = [c.strip() for c in line.replace('COLUMNS:', '').split(',')]
                    elif line.startswith('TYPE:'):
                        viz_type = line.replace('TYPE:', '').strip().lower()
                
                if columns and viz_type and all(col in df.columns for col in columns):
                    if len(columns) == 1:
                        viz_data = {
                            'column': columns[0],
                            'type': viz_type,
                            'title': f"{viz_type.capitalize()} of {columns[0]}"
                        }
                        # Use create_visualization for single column visualizations
                        create_visualization(df, columns[0], viz_type, viz_data['title'])
                    else:
                        viz_data = {
                            'columns': columns,
                            'type': viz_type,
                            'title': f"{viz_type.capitalize()} of {columns[0]} vs {columns[1]}"
                        }
                        # Use create_multi_column_viz for multiple columns
                        create_multi_column_viz(df, columns, viz_type)
    
                    if 'visualization_history' in st.session_state:
                        st.session_state.visualization_history.append(viz_data)
                        # Keep only last 5 visualizations
                        if len(st.session_state.visualization_history) > 5:
                            st.session_state.visualization_history = st.session_state.visualization_history[-5:]
                        
            except Exception as viz_error:
                st.error(f"Visualization error: {str(viz_error)}")

        if "histogram" in question.lower() or "distribution" in question.lower():
            column_query = _pandas_agent.run(f"""
            For the question: "{question}"
            1. Identify which numeric column needs a histogram
            2. Return ONLY the column name in this format:
            COLUMN: <column_name>
            """)
            
            try:
                column = column_query.split('COLUMN:')[1].strip()
                if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                    stats = _pandas_agent.run(f"""
                    Analyze the distribution of {column} and provide:
                    1. The range of values
                    2. The mean and median
                    3. Any notable patterns or skewness
                    Keep it concise.
                    """)
                    
                    viz_data = {
                        'column': column,
                        'type': 'histogram',
                        'title': f"Distribution of {column}",
                        'stats': stats
                    }
                    create_visualization(df, column, 'histogram', f"Distribution of {column}")
                    
                    if 'visualization_history' in st.session_state:
                        st.session_state.visualization_history.append(viz_data)
                    
            except Exception as e:
                st.error(f"Could not create histogram. Error: {str(e)}")

        # Generate final response
        analysis_prompt = f"""
        Previous conversation context (last 5 exchanges):
        {context_str}
        
        Current question: {question}
        
        When analyzing:
        1. Consider relationships between relevant columns
        2. If finding specific values, include their related data
        3. Process complete rows of data when needed
        4. Include supporting evidence in the response
        
        Execute the analysis and return complete results.
        """
        
        response = _pandas_agent.run(analysis_prompt if not needs_viz else question)
        return response, viz_data
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return f"I encountered an error while processing your question: {str(e)}", None

# Cache initial analysis
@st.cache_data(show_spinner="Performing initial analysis...")
def initial_analysis(_pandas_agent, df):
    """Perform initial EDA and store results in session state"""
    if not st.session_state.analysis_complete:
        st.write("Data Overview")
        st.write("The first rows of your dataset look like this:")
        st.write(df.head())
        
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        st.write("\n*Categorical Columns Available:*")
        st.write(list(categorical_columns))
        
        st.write("Data Quality Assessment")
        columns_df = _pandas_agent.run("""For each column, provide:
        1. The data type
        2. Whether it's categorical or numerical
        3. A brief description of what the column represents
        """)
        st.write(columns_df)
        
        missing_values = _pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
        st.write(missing_values)
        
        duplicates = _pandas_agent.run("Are there any duplicate values and if so where?")
        st.write(duplicates)
        
        st.session_state.analysis_complete = True
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "I've completed the initial analysis. What would you like to know about your data? You can ask me anything about the patterns, relationships, or specific aspects of your dataset."
        })

def create_multi_column_viz(df, columns, viz_type):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if viz_type == "scatter":
            ax.scatter(df[columns[0]], df[columns[1]], alpha=0.5)
        elif viz_type == "bar":
            # For categorical vs numerical comparisons
            grouped_data = df.groupby(columns[0])[columns[1]].mean().sort_values(ascending=False)
            grouped_data.plot(kind='bar', ax=ax)
        elif viz_type == "line":
            df.sort_values(columns[0]).plot(x=columns[0], y=columns[1], kind='line', ax=ax)
        elif viz_type == "box":
            df.boxplot(column=columns[1], by=columns[0], ax=ax)
            
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close()
        return True
    except Exception as e:
        st.error(f"Error in multi-column visualization: {str(e)}")
        return False

# OpenAIKey
os.environ['OPENAI_API_KEY']=st.secrets["openai_apikey"]
load_dotenv(find_dotenv())

# Title
st.title('SU iBot 🤖')

# Welcoming message
st.write("Hello, 👋 I am your AI Assistant and I am here to help you with your data science projects.")

# Explanation sidebar
with st.sidebar:
    st.write('Your Data Science Adventure Begins with an CSV File.')
    st.caption('''**You may already know that every exciting data science journey starts with a dataset.
    That's why I'd love for you to upload a CSV file.
    Once we have your data in hand, we'll dive into understanding it and have some fun exploring it.
    Then, we'll work together to shape your business challenge into a data science framework.**
    ''')
    st.divider()
    st.caption("<p style='text-align:center'> made with ❤ by SU iBot</p>", unsafe_allow_html=True)

# Initialize session state
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'visualization_history' not in st.session_state:
    st.session_state.visualization_history = []
# Optional: Add a max history size
if 'max_history' not in st.session_state:
    st.session_state.max_history = 5

# Function to manage chat history size
def add_to_chat_history(message):
    st.session_state.chat_history.append(message)
    # Keep only the last N messages
    if len(st.session_state.chat_history) > st.session_state.max_history * 2:  # *2 to account for both user and assistant messages
        st.session_state.chat_history = st.session_state.chat_history[-st.session_state.max_history * 2:]

def store_visualization(column, viz_type, title, stats=None):
    """Store visualization metadata in session state"""
    viz_data = {
        'column': column,
        'type': viz_type,
        'title': title,
        'stats': stats,
        'timestamp': pd.Timestamp.now()
    }
    st.session_state.visualization_history.append(viz_data)

def display_chat_message(role, content, with_visualization=None):
    """Display a chat message with optional visualization"""
    with st.chat_message(role):
        st.write(content)
        if with_visualization:
            # Check if it's a single column or multi-column visualization
            if 'columns' in with_visualization:
                # Multi-column visualization
                create_multi_column_viz(
                    df, 
                    with_visualization['columns'],
                    with_visualization['type']
                )
            elif 'column' in with_visualization:
                # Single column visualization
                create_visualization(
                    df, 
                    with_visualization['column'],
                    with_visualization['type'],
                    with_visualization['title']
                )
            
            if with_visualization.get('stats'):
                st.write("Distribution Insights:")
                st.write(with_visualization['stats'])
def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click=clicked, args=[1])
if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your file here", type="csv")
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)

        # Create LLM and pandas agent using cached functions
        llm = get_llm()
        pandas_agent = get_pandas_agent(llm, df)

        # Perform initial analysis
        initial_analysis(pandas_agent, df)

        # Chat interface
        st.write("---")
        st.subheader("Ask me anything about your data 💭")
        
        # Display full chat history with visualizations
        for message in st.session_state.chat_history:
            display_chat_message(
                message["role"],
                message["content"],
                message.get("visualization")
            )

        # User input
        user_question = st.chat_input("Type your question here...")
        if user_question:
            # Add user message to chat history using the new function
            add_to_chat_history({
                "role": "user",
                "content": user_question
            })
            # Display user message
            display_chat_message("user", user_question)
            
            
            # Process the question and get response
            with st.chat_message("assistant"):
                response, viz_data = process_question(pandas_agent, user_question, df)
                st.write(response)
                
                # Add assistant's response to chat history
                message_data = {
                    "role": "assistant",
                    "content": response,
                }
                if viz_data:
                    message_data["visualization"] = viz_data
                
                add_to_chat_history(message_data)
               