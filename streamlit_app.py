import streamlit as st
from youtube_agent import YouTubeAgent
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
from collections import defaultdict
import re
from streamlit_plotly_events import plotly_events
import json
import os
import pickle
import sys
import logging
import hashlib
import requests
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Print to terminal
        logging.FileHandler('youtube_organizer.log')  # Save to file
    ]
)
logger = logging.getLogger(__name__)

# Add a startup log message
logger.info("Application starting...")

def format_video_card(video):
    """Creates a formatted card-like display for a video"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        thumbnail_url = f"https://img.youtube.com/vi/{video['video_id']}/mqdefault.jpg"
        st.image(thumbnail_url, use_container_width=True)
    
    with col2:
        st.markdown(f"### [{video['title']}](https://youtube.com/watch?v={video['video_id']})")
        st.write(f"**Channel:** {video['channel_title']}")
        
        # Handle datetime properly
        if isinstance(video['published_at'], pd.Timestamp):
            published_date = video['published_at'].strftime('%Y-%m-%d')
        else:
            published_date = video['published_at'][:10]
        st.write(f"**Published:** {published_date}")

def detect_language_with_llama(text, description="", show_llm_details=False):
    """Use locally running Llama 3.1 via Ollama for language detection"""
    try:
        # Combine title and description for better context
        full_text = f"Title: {text}\nDescription: {description}"
        
        # Craft a more direct prompt for language detection
        prompt = f"""Analyze this YouTube video content and determine its primary language. You must respond in this exact format:
Language: [ONLY one of these: English, Hindi, Tamil, Telugu, Malayalam, Kannada, Marathi, Bengali, Punjabi, Gujarati, Bhojpuri, Odia, Assamese, Urdu, Korean, Japanese, Chinese, Arabic, Spanish, French, German, Other]
Reason: [brief explanation in 1-2 sentences]

Content to analyze:
{full_text}"""

        # Log LLM request details
        logger.info("\n" + "="*80)
        logger.info("LLM REQUEST")
        logger.info("="*80)
        logger.info(f"Video Title: {text}")
        logger.info(f"Description: {description}")
        logger.info("-"*80)
        logger.info("Prompt:")
        logger.info(prompt)
        logger.info("="*80)

        # Call Ollama API with Llama 3.1
        response = requests.post('http://localhost:11434/api/generate',
            json={
                "model": "llama3.1",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 128,
                    "stop": ["\n\n"],
                    "num_ctx": 2048
                }
            },
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '').strip()
            
            # Parse language and reason from response
            try:
                # Extract language and reason using more robust parsing
                lines = response_text.split('\n')
                language_line = next((line for line in lines if line.startswith('Language:')), '')
                reason_line = next((line for line in lines if line.startswith('Reason:')), '')
                
                detected_language = language_line.replace('Language:', '').strip()
                reason = reason_line.replace('Reason:', '').strip()
                
                # Validate language is in our list
                valid_languages = {'English', 'Hindi', 'Tamil', 'Telugu', 'Malayalam', 'Kannada', 
                                 'Marathi', 'Bengali', 'Punjabi', 'Gujarati', 'Bhojpuri', 'Odia', 
                                 'Assamese', 'Urdu', 'Korean', 'Japanese', 'Chinese', 'Arabic', 
                                 'Spanish', 'French', 'German', 'Other'}
                
                if detected_language not in valid_languages:
                    detected_language = 'Other'
                    reason = 'Could not confidently determine the language'
                
            except Exception as e:
                logger.error(f"Error parsing LLM response: {str(e)}")
                detected_language = 'Other'
                reason = 'Error parsing language detection response'
            
            # Log LLM response details
            logger.info("\nLLM RESPONSE")
            logger.info("="*80)
            logger.info(f"Raw Response: {result}")
            logger.info(f"Detected Language: {detected_language}")
            logger.info(f"Reason: {reason}")
            logger.info("="*80 + "\n")
            
            return detected_language, reason
            
    except Exception as e:
        logger.error(f"Error in Llama language detection: {str(e)}")
        return "Other", "Error in language detection"

def detect_language_basic(text):
    """Fallback basic language detection using patterns"""
    # Original pattern-based detection code here
    patterns = {
        'Korean': r'[ㄱ-ㅎㅏ-ㅣ가-힣]',
        'Japanese': r'[ぁ-んァ-ン一-龥]',
        'Hindi': r'[ऀ-ॿ]|hindi|bollywood',
        'Tamil': r'[஀-௺]|tamil|தமிழ்',
        'Telugu': r'[ఀ-౿]|telugu|తెలుగు',
        'Malayalam': r'[ഀ-ൿ]|malayalam|മലയാളം',
        'Kannada': r'[ಀ-೿]|kannada|ಕನ್ನಡ',
        'Arabic': r'[\u0600-\u06FF]',
        'Chinese': r'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]',
    }
    
    text = f"{text}".lower()
    
    # Check for common Indian music indicators first
    if any(x in text for x in ['indian', 'desi', 'bhajan', 'devotional', 'carnatic', 'hindustani']):
        # Then check specific language indicators
        if any(x in text for x in ['hindi', 'bollywood']):
            return 'Hindi'
        elif 'tamil' in text or 'தமிழ்' in text:
            return 'Tamil'
        elif 'telugu' in text or 'తెలుగు' in text:
            return 'Telugu'
        elif 'malayalam' in text or 'മലയാളം' in text:
            return 'Malayalam'
        elif 'kannada' in text or 'ಕನ್ನಡ' in text:
            return 'Kannada'
        return 'Indian (Other)'
    
    # Check script patterns
    for lang, pattern in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return lang
            
    # Additional checks for Indian languages based on channel names
    indian_channels = [
        'T-Series', 'Zee Music', 'Sony Music India', 'Saregama', 
        'Think Music', 'Sun Music', 'Aditya Music', 'Lahari Music',
        'Manorama Music', 'Malayalam Songs', 'Tamil Songs', 'Telugu Songs'
    ]
    
    if any(channel.lower() in text for channel in indian_channels):
        # Try to determine specific language based on channel name
        channel_lower = text.lower()
        if any(x in channel_lower for x in ['tamil', 'sun music']):
            return 'Tamil'
        elif any(x in channel_lower for x in ['telugu', 'aditya music']):
            return 'Telugu'
        elif any(x in channel_lower for x in ['malayalam', 'manorama']):
            return 'Malayalam'
        elif any(x in channel_lower for x in ['kannada', 'anand audio']):
            return 'Kannada'
        elif any(x in channel_lower for x in ['hindi', 't-series', 'zee music']):
            return 'Hindi'
        return 'Indian (Other)'
    
    # Default to English/Others if no specific patterns found
    return 'English/Others'

def detect_genre(title, channel_title):
    """Simple genre detection based on keywords"""
    genres = {
        'Pop': ['pop', 'top hits', 'billboard'],
        'Rock': ['rock', 'metal', 'alternative'],
        'Hip Hop': ['rap', 'hip hop', 'trap'],
        'Electronic': ['edm', 'electronic', 'dance', 'dj'],
        'Classical': ['classical', 'orchestra', 'symphony'],
        'Jazz': ['jazz', 'blues'],
        'K-pop': ['k-pop', 'kpop'],
        'J-pop': ['j-pop', 'jpop']
    }
    
    text = f"{title} {channel_title}".lower()
    
    for genre, keywords in genres.items():
        if any(keyword in text for keyword in keywords):
            return genre
            
    return 'Other'

def detect_video_category(title, channel_title, confidence_threshold=0.6):
    """
    Detect category for non-music videos with confidence score
    Returns tuple of (category, confidence)
    """
    categories = {
        'Gaming': {
            'keywords': ['gameplay', 'gaming', 'playthrough', 'xbox', 'playstation', 'nintendo'],
            'strong_indicators': ['gameplay', 'gaming', 'playthrough'],
            'channels': ['IGN', 'GameSpot', 'PlayStation', 'Xbox', 'Nintendo']
        },
        'Technology': {
            'keywords': ['tech', 'coding', 'programming', 'software', 'hardware', 'review'],
            'strong_indicators': ['coding tutorial', 'programming', 'tech review'],
            'channels': ['MKBHD', 'Linus Tech Tips', 'Verge', 'TechCrunch']
        },
        'Education': {
            'keywords': ['learn', 'tutorial', 'course', 'education', 'training', 'lecture'],
            'strong_indicators': ['course', 'lecture', 'learning'],
            'channels': ['Coursera', 'Khan Academy', 'MIT OpenCourseWare']
        },
        'News & Politics': {
            'keywords': ['news', 'politics', 'current affairs', 'report', 'coverage'],
            'strong_indicators': ['breaking news', 'live coverage', 'official news'],
            'channels': ['CNN', 'BBC News', 'Reuters', 'AP News']
        }
    }
    
    text = f"{title} {channel_title}".lower()
    
    # Calculate confidence scores for each category
    category_scores = {}
    
    for category, rules in categories.items():
        score = 0
        total_checks = 3  # Number of criteria we're checking
        
        # Check keywords
        keyword_matches = sum(1 for keyword in rules['keywords'] if keyword in text)
        if keyword_matches > 0:
            score += (keyword_matches / len(rules['keywords']))
        
        # Check strong indicators (higher weight)
        strong_matches = sum(1 for indicator in rules['strong_indicators'] if indicator in text)
        if strong_matches > 0:
            score += (strong_matches / len(rules['strong_indicators'])) * 2
        
        # Check channel names (highest weight)
        channel_matches = sum(1 for channel in rules['channels'] if channel.lower() in channel_title.lower())
        if channel_matches > 0:
            score += 1
        
        # Calculate final confidence score
        category_scores[category] = score / total_checks

    # Get category with highest confidence
    if category_scores:
        best_category = max(category_scores.items(), key=lambda x: x[1])
        if best_category[1] >= confidence_threshold:
            return best_category[0], best_category[1]
    
    return "Uncategorized", 0.0

def check_ollama_connection():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = response.json()
            if any(model.get('name', '').startswith('llama3.1') for model in models.get('models', [])):
                logger.info("Successfully connected to Ollama with Llama 3.1 model")
                return True
            else:
                logger.error("Llama 3.1 model not found in Ollama")
                return False
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {str(e)}")
        return False

def analyze_videos(videos, is_music=True):
    """Analyze videos with error handling and data validation"""
    logger.info(f"Analyzing {'music' if is_music else 'other'} videos")
    
    # Add timestamp for unique chart keys
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    try:
        # Validate input data
        if not videos:
            logger.error("No videos provided for analysis")
            st.error("No videos available for analysis")
            return pd.DataFrame()
            
        # Ensure videos is a list of dictionaries
        if not isinstance(videos, list):
            logger.error(f"Expected list of videos, got {type(videos)}")
            st.error("Invalid video data format")
            return pd.DataFrame()
            
        # Create DataFrame with error handling
        try:
            df = pd.DataFrame(videos)
            required_columns = ['title', 'video_id', 'channel_title', 'published_at']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                st.error(f"Video data missing required information: {', '.join(missing_columns)}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error creating DataFrame: {str(e)}")
            st.error("Could not process video data")
            return pd.DataFrame()
            
        # Process dates with error handling
        try:
            df['published_at'] = pd.to_datetime(df['published_at'])
            df['year'] = df['published_at'].dt.year
        except Exception as e:
            logger.error(f"Error processing dates: {str(e)}")
            st.warning("Could not process some date information")
        
        if is_music:
            # First load the page with basic detection for all videos
            try:
                df['language'] = df.apply(lambda x: detect_language_basic(x['title']), axis=1)
                df['genre'] = df.apply(lambda x: detect_genre(x['title'], x['channel_title']), axis=1)
            except Exception as e:
                logger.error(f"Error in basic language/genre detection: {str(e)}")
                st.warning("Some language or genre detection may be incomplete")
            
            # Show normal analytics first
            try:
                display_analytics(df)
            except Exception as e:
                logger.error(f"Error displaying analytics: {str(e)}")
                st.error("Could not display some analytics")
            
            # LLM Analysis only for English/Others videos
            st.markdown("---")
            st.header("Language Analysis with LLM")
            
            # Get English/Others videos
            english_others_df = df[df['language'] == 'English/Others']
            
            if not english_others_df.empty:
                st.info(f"Analyzing {len(english_others_df)} videos from English/Others category")
                
                # Create table to store results
                analysis_results = []
                
                # Process each video
                for _, video in english_others_df.iterrows():
                    try:
                        # Check cache first
                        cached_lang, cached_reason = get_cached_llm_response(video['title'])
                        
                        if cached_lang and cached_reason:
                            # Use cached result
                            llm_lang, reason = cached_lang, cached_reason
                        else:
                            # Get LLM detection with reasoning
                            llm_lang, reason = detect_language_with_llama(
                                video['title'],
                                video.get('description', ''),
                                show_llm_details=True
                            )
                            # Cache the result
                            cache_llm_response(video['title'], llm_lang, reason)
                        
                        # Store result
                        analysis_results.append({
                            'Title': video['title'],
                            'Channel': video['channel_title'],
                            'Initial Category': 'English/Others',
                            'LLM Detection': llm_lang,
                            'Reasoning': reason,
                            'Published': video['published_at'].strftime('%Y-%m-%d')
                        })
                        
                    except Exception as e:
                        logger.error(f"Error in LLM analysis: {str(e)}")
                        continue
                
                # Show results table
                if analysis_results:
                    results_df = pd.DataFrame(analysis_results)
                    
                    # Format long text columns for better readability
                    def format_long_text(text, max_length=50):
                        """Split long text into multiple lines"""
                        words = text.split()
                        lines = []
                        current_line = []
                        current_length = 0
                        
                        for word in words:
                            if current_length + len(word) + 1 <= max_length:
                                current_line.append(word)
                                current_length += len(word) + 1
                            else:
                                lines.append(' '.join(current_line))
                                current_line = [word]
                                current_length = len(word)
                        
                        if current_line:
                            lines.append(' '.join(current_line))
                        
                        return '\n'.join(lines)
                    
                    # Format the columns
                    results_df['Title'] = results_df['Title'].apply(lambda x: format_long_text(x, max_length=40))
                    results_df['Reasoning'] = results_df['Reasoning'].apply(lambda x: format_long_text(x, max_length=50))
                    
                    # Display the formatted table
                    st.dataframe(
                        results_df,
                        column_config={
                            'Title': st.column_config.TextColumn(
                                'Video Title',
                                width=250,
                                help="Title of the video",
                                default="...",
                            ),
                            'Channel': st.column_config.TextColumn(
                                'Channel',
                                width=150,
                                help="YouTube channel"
                            ),
                            'Initial Category': st.column_config.TextColumn(
                                'Initial Category',
                                width=120,
                                help="Initial language category"
                            ),
                            'LLM Detection': st.column_config.TextColumn(
                                'Detected Language',
                                width=120,
                                help="Language detected by LLM"
                            ),
                            'Reasoning': st.column_config.TextColumn(
                                'LLM Reasoning',
                                width=250,
                                help="LLM's reasoning for language detection",
                                default="..."
                            ),
                            'Published': st.column_config.TextColumn(
                                'Published Date',
                                width=100,
                                help="Video publication date"
                            ),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Add expandable section to show full details
                    with st.expander("Show Full Details"):
                        for _, row in results_df.iterrows():
                            st.markdown(f"""
                            **Title:** {row['Title']}  
                            **Channel:** {row['Channel']}  
                            **Language:** {row['LLM Detection']}  
                            **Reasoning:** {row['Reasoning']}  
                            **Published:** {row['Published']}
                            ---
                            """)
                    
                    # Create updated language distribution
                    st.subheader("Updated Language Distribution")

                    # Create a copy of original DataFrame
                    updated_df = df.copy()

                    # Create mapping from results DataFrame, not analysis_results list
                    results_df = pd.DataFrame(analysis_results)
                    llm_detections = dict(zip(results_df['Title'], results_df['LLM Detection']))

                    # Update languages for English/Others videos
                    mask = updated_df['language'] == 'English/Others'
                    updated_df.loc[mask, 'language'] = updated_df.loc[mask, 'title'].map(llm_detections).fillna('Other')

                    # Calculate the updated distribution
                    updated_lang_counts = updated_df['language'].value_counts()

                    col1, col2 = st.columns(2)

                    with col1:
                        # Create pie chart for updated distribution
                        chart_key = f"updated_language_pie_chart_{timestamp}_{random.randint(1000, 9999)}"
                        
                        # Create DataFrame for the pie chart
                        pie_data = pd.DataFrame({
                            'Language': updated_lang_counts.index,
                            'Count': updated_lang_counts.values,
                            'Percentage': (updated_lang_counts.values / len(updated_df) * 100).round(1)
                        })
                        
                        fig_updated = px.pie(
                            pie_data,
                            names='Language',
                            values='Count',
                            title="After LLM Analysis",
                            hole=0.3,
                            color_discrete_sequence=px.colors.qualitative.Set3,
                            hover_data=['Percentage'],
                            custom_data=['Percentage']
                        )
                        
                        fig_updated.update_layout(
                            showlegend=True,
                            height=400,
                            margin=dict(t=0, b=0, l=0, r=0),
                            title=dict(
                                text="After LLM Analysis",
                                y=0.95,
                                x=0.5,
                                xanchor='center',
                                yanchor='top'
                            )
                        )
                        
                        # Update hover template to show percentage
                        fig_updated.update_traces(
                            hovertemplate="<b>%{label}</b><br>" +
                                         "Count: %{value}<br>" +
                                         "Percentage: %{customdata[0]:.1f}%<extra></extra>"
                        )
                        
                        fig_updated.update_layout(
                            showlegend=True, 
                            height=400,
                            title_x=0.5
                        )
                        st.plotly_chart(fig_updated, use_container_width=True, key=chart_key)

                    with col2:
                        # Show detailed stats
                        st.markdown("### Language Distribution Changes")
                        
                        # Create comparison DataFrame
                        original_counts = df['language'].value_counts()
                        comparison_df = pd.DataFrame({
                            'Original': original_counts,
                            'Updated': updated_lang_counts
                        }).fillna(0).astype(int)
                        
                        # Calculate changes
                        comparison_df['Change'] = comparison_df['Updated'] - comparison_df['Original']
                        
                        # Format for display without background gradient
                        st.dataframe(
                            comparison_df.style.format({
                                'Original': '{:,}',
                                'Updated': '{:,}',
                                'Change': lambda x: f"+{x:,}" if x > 0 else f"{x:,}"  # Show + sign for positive changes
                            }),
                            use_container_width=True
                        )
                        
                        # Show summary
                        st.info(f"""
                        Analysis Summary:
                        - Original English/Others videos: {len(df[df['language'] == 'English/Others'])}
                        - Reclassified into:
                          {', '.join(f"{lang}: {count}" for lang, count in results_df['LLM Detection'].value_counts().items())}
                        
                        Total videos by language after reclassification:
                        {', '.join(f"{lang}: {count}" for lang, count in updated_lang_counts.items())}
                        """)
                
                # Add side-by-side comparison of language distributions
                st.markdown("---")
                st.subheader("Language Distribution Comparison")
                st.markdown("<br>", unsafe_allow_html=True)  # Add space after subheader

                comparison_col1, comparison_col2 = st.columns(2)

                with comparison_col1:
                    # Original Language Distribution
                    fig_original = px.pie(
                        names=df['language'].value_counts().index,
                        values=df['language'].value_counts().values,
                        title="Initial Language Distribution",
                        hole=0.3,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_original.update_layout(
                        showlegend=True,
                        height=400,
                        margin=dict(t=50, b=0, l=0, r=0),  # Increased top margin
                        title=dict(
                            text="Initial Language Distribution",
                            y=0.98,  # Moved title up
                            x=0.5,
                            xanchor='center',
                            yanchor='top',
                            font=dict(size=16)  # Increased font size
                        )
                    )
                    chart_key = f"comparison_original_{timestamp}_{random.randint(1000, 9999)}"
                    st.plotly_chart(fig_original, use_container_width=True, key=chart_key)

                with comparison_col2:
                    # Updated Language Distribution
                    fig_final = px.pie(
                        names=updated_lang_counts.index,
                        values=updated_lang_counts.values,
                        title="After LLM Analysis",
                        hole=0.3,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_final.update_layout(
                        showlegend=True,
                        height=400,
                        margin=dict(t=50, b=0, l=0, r=0),  # Increased top margin
                        title=dict(
                            text="After LLM Analysis",
                            y=0.98,  # Moved title up
                            x=0.5,
                            xanchor='center',
                            yanchor='top',
                            font=dict(size=16)  # Increased font size
                        )
                    )
                    chart_key = f"comparison_final_{timestamp}_{random.randint(1000, 9999)}"
                    st.plotly_chart(fig_final, use_container_width=True, key=chart_key)

                # Add summary of changes
                st.info("""
                **Key Changes in Distribution:**
                - English/Others videos have been reclassified into specific languages
                - More accurate language identification for ambiguous cases
                - Better representation of actual language diversity in the collection
                """)
            else:
                st.warning("No LLM analysis results available")
        else:
            st.info("No videos found in English/Others category")
        
        return df
        
    except Exception as e:
        logger.error(f"Unexpected error in analyze_videos: {str(e)}")
        st.error("An error occurred while analyzing videos")
        return pd.DataFrame()

def display_analytics(df):
    """Display analytics for music videos"""
    # Add timestamp and random component to make keys unique
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    st.header("Music Analytics")
    
    total_videos = len(df)
    st.subheader(f"Total Music Videos: {total_videos}")

    if total_videos == 0:
        st.warning("No music videos to analyze")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Language Distribution
        st.subheader("Language Distribution")
        lang_counts = df['language'].value_counts()
        
        if not lang_counts.empty:
            # Create pie chart with unique key
            fig_lang = px.pie(
                names=lang_counts.index,
                values=lang_counts.values,
                hole=0.3,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_lang.update_layout(
                showlegend=True,
                height=400,
                margin=dict(t=0, b=0, l=0, r=0)
            )
            # Add random component to key
            chart_key = f"language_pie_chart_{timestamp}_{random.randint(1000, 9999)}"
            st.plotly_chart(fig_lang, use_container_width=True, key=chart_key)
            
            # Show detailed language stats
            lang_df = pd.DataFrame({
                'Language': lang_counts.index,
                'Videos': lang_counts.values,
                'Percentage': (lang_counts.values / total_videos * 100).round(1)
            }).sort_values('Videos', ascending=False)
            
            st.dataframe(
                lang_df.style.format({
                    'Percentage': '{:.1f}%',
                    'Videos': '{:,}'
                }),
                hide_index=True,
                use_container_width=True
            )
            
            # Add language insights
            st.markdown("### Language Insights")
            primary_lang = lang_counts.index[0]
            secondary_lang = lang_counts.index[1] if len(lang_counts) > 1 else None
            
            st.write(f"• Primary language: **{primary_lang}** ({lang_df.iloc[0]['Percentage']}% of videos)")
            if secondary_lang:
                st.write(f"• Secondary language: **{secondary_lang}** ({lang_df.iloc[1]['Percentage']}% of videos)")
            
            # Show language diversity
            diversity_score = (1 - (lang_counts.values[0] / total_videos)) * 100
            st.write(f"• Language diversity score: **{diversity_score:.1f}%**")
        else:
            st.warning("No language data available")

    with col2:
        # Genre Distribution
        st.subheader("Genre Distribution")
        genre_counts = df['genre'].value_counts()
        
        if not genre_counts.empty:
            # Create pie chart with unique key
            fig_genre = px.pie(
                names=genre_counts.index,
                values=genre_counts.values,
                hole=0.3,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_genre.update_layout(
                showlegend=True,
                height=400,
                margin=dict(t=0, b=0, l=0, r=0)
            )
            # Add random component to key
            chart_key = f"genre_pie_chart_{timestamp}_{random.randint(1000, 9999)}"
            st.plotly_chart(fig_genre, use_container_width=True, key=chart_key)
            
            # Show genre stats
            genre_df = pd.DataFrame({
                'Genre': genre_counts.index,
                'Videos': genre_counts.values,
                'Percentage': (genre_counts.values / total_videos * 100).round(1)
            })
            st.dataframe(
                genre_df.style.format({'Percentage': '{:.1f}%'}),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("No genre data available")

    # Timeline Analysis with unique key
    st.subheader("Videos by Year")
    yearly_counts = df['year'].value_counts().sort_index()
    
    fig_timeline = px.bar(
        x=yearly_counts.index,
        y=yearly_counts.values,
        labels={'x': 'Year', 'y': 'Number of Videos'},
        color_discrete_sequence=['#2ecc71']
    )
    fig_timeline.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Videos",
        showlegend=False,
        height=400,
        margin=dict(t=20, b=40, l=40, r=20)
    )
    fig_timeline.update_traces(
        texttemplate='%{y}',
        textposition='outside'
    )
    # Add random component to key
    chart_key = f"timeline_bar_chart_{timestamp}_{random.randint(1000, 9999)}"
    st.plotly_chart(fig_timeline, use_container_width=True, key=chart_key)

def display_other_analytics(df):
    """Display analytics for non-music videos"""
    st.header("Other Videos Analytics")
    
    # Show only confidently categorized videos
    confident_df = df[df['category_confidence'] >= 0.6]
    uncertain_df = df[df['category_confidence'] < 0.6]
    
    st.write(f"Confidently categorized videos: {len(confident_df)}")
    st.write(f"Uncategorized videos: {len(uncertain_df)}")

    # Show category distribution for confident categories
    if not confident_df.empty:
        st.subheader("Category Distribution (Confident Categories)")
        category_counts = confident_df['category'].value_counts()
        
        fig_category = px.pie(
            names=category_counts.index,
            values=category_counts.values,
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig_category, use_container_width=True)
        
        # Show category stats
        category_df = pd.DataFrame({
            'Category': category_counts.index,
            'Videos': category_counts.values,
            'Percentage': (category_counts.values / len(confident_df) * 100).round(1)
        })
        st.dataframe(category_df, hide_index=True)

    # Add filtering with confidence indicator
    st.subheader("Filter Videos")
    show_uncertain = st.checkbox("Show uncategorized videos")
    
    display_df = df if show_uncertain else confident_df
    categories = ['All'] + list(display_df['category'].unique())
    selected_category = st.selectbox("Select Category", categories)
    
    if selected_category != 'All':
        filtered_df = display_df[display_df['category'] == selected_category]
    else:
        filtered_df = display_df

    # Display filtered videos with confidence scores
    if not filtered_df.empty:
        st.subheader(f"Filtered Videos ({len(filtered_df)} videos)")
        for _, video in filtered_df.iterrows():
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    format_video_card(video)
                with col2:
                    if 'category_confidence' in video:
                        confidence = video['category_confidence'] * 100
                        st.progress(confidence / 100)
                        st.caption(f"Category confidence: {confidence:.1f}%")
                st.divider()

def save_to_cache(data):
    """Save data to cache file"""
    cache_file = "youtube_cache.json"
    try:
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'data': {
                'liked_videos': data['liked_videos'],
                'music_videos': data['music_videos'],
                'other_videos': data['other_videos']
            }
        }
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        logger.info("Cache saved successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving cache: {e}")
        return False

def load_from_cache():
    """Load data from cache if it exists and is not expired"""
    cache_file = "youtube_cache.json"
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                
                # Check if cache is less than 24 hours old
                if datetime.now() - cache_time < timedelta(hours=24):
                    logger.info("Valid cache found and loaded")
                    return cache_data['data']
        logger.info("No valid cache found")
        return None
    except Exception as e:
        logger.error(f"Error loading cache: {e}")
        return None

def clear_cache():
    """Clear the cache file"""
    cache_file = "youtube_cache.json"
    try:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            return True
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
    return False

def create_youtube_tile(video_info):
    """Create a YouTube-style video tile with preview, play option, and detailed reason"""
    try:
        video_id = video_info.get('video_id', '')
        title = video_info.get('title', '').replace('"', '&quot;')  # Escape quotes
        channel = video_info.get('channel', '').replace('"', '&quot;')
        reason = video_info.get('reason', '').replace('\n', '<br>')  # Convert newlines to HTML breaks
        
        # Create a single, properly formatted HTML string
        html = f'''
            <div style="
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                background: white;
                margin-bottom: 16px;
                transition: transform 0.2s ease;
            ">
                <a href="https://youtube.com/watch?v={video_id}" target="_blank" style="text-decoration: none; color: inherit;">
                    <div style="position: relative; width: 100%; padding-top: 56.25%;">
                        <img 
                            src="https://i.ytimg.com/vi/{video_id}/mqdefault.jpg"
                            style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover;"
                            onerror="this.onerror=null; this.src='https://placehold.co/320x180/f0f0f0/666666.png?text=No+Preview'"
                        />
                    </div>
                    <div style="padding: 12px;">
                        <div style="
                            font-weight: 500;
                            font-size: 14px;
                            margin-bottom: 4px;
                            overflow: hidden;
                            display: -webkit-box;
                            -webkit-line-clamp: 2;
                            -webkit-box-orient: vertical;
                        ">{title}</div>
                        <div style="color: #606060; font-size: 12px;">{channel}</div>
                    </div>
                </a>
                <div style="
                    background-color: #f8f9fa;
                    padding: 12px;
                    margin: 0 12px 12px 12px;
                    border-radius: 4px;
                    font-size: 12px;
                    color: #666;
                    border-left: 3px solid #ff0000;
                ">
                    <div style="font-weight: 500; margin-bottom: 4px;">Why we recommend this:</div>
                    <div style="line-height: 1.4;">{reason}</div>
                </div>
            </div>
        '''
        
        # Render the HTML
        st.markdown(html, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error creating YouTube tile: {str(e)}")
        st.error("Error loading video preview")

def get_authenticated_youtube():
    """Get authenticated YouTube API client"""
    try:
        agent = YouTubeAgent()
        agent.authenticate()
        return agent.youtube
    except Exception as e:
        logger.error(f"Error authenticating YouTube: {str(e)}")
        return None

def get_related_videos(youtube, video_id, max_results=5):
    """Fetch related videos using YouTube API with content-based filtering"""
    if youtube is None:
        logger.error("YouTube client is not authenticated")
        return []
        
    try:
        # First get video details
        video_request = youtube.videos().list(
            part="snippet,topicDetails,contentDetails",
            id=video_id
        ).execute()
        
        if not video_request.get('items'):
            return []
            
        source_video = video_request['items'][0]['snippet']
        source_channel_id = source_video['channelId']
        source_title = source_video['title'].lower()
        source_tags = source_video.get('tags', [])
        
        # Get source video details
        source_language = detect_language_with_llama(source_title, source_video['channelTitle'])
        source_genre = detect_genre(source_title, source_video['channelTitle'])
        
        # Extract artists from source title
        source_artists = extract_artists(source_title, source_video['channelTitle'])
        
        # Search for similar videos
        search_query = ' '.join([
            source_video.get('categoryId', ''),
            ' '.join(source_tags[:5] if source_tags else []),
            source_language,
            source_genre,
            ' '.join(source_artists)
        ])
        
        request = youtube.search().list(
            part="snippet",
            q=search_query,
            type="video",
            maxResults=max_results * 2,
            videoCategoryId=source_video.get('categoryId'),
            relevanceLanguage=source_language[:2],
            order="relevance"
        )
        response = request.execute()
        
        related_videos = []
        for item in response.get('items', []):
            if (item['id']['kind'] == 'youtube#video' and 
                item['id']['videoId'] != video_id and
                item['snippet']['channelId'] != source_channel_id):
                
                video_title = item['snippet']['title']
                video_channel = item['snippet']['channelTitle']
                
                # Build detailed recommendation reasons
                reasons = []
                
                # Check language and add specific reason
                video_language = detect_language_with_llama(video_title, video_channel)
                if video_language == source_language:
                    reasons.append(f"• Features {video_language} music like your other favorites")
                
                # Check genre and add specific reason
                video_genre = detect_genre(video_title, video_channel)
                if video_genre == source_genre:
                    reasons.append(f"• Matches your interest in {video_genre} music")
                
                # Check for similar artists
                video_artists = extract_artists(video_title, video_channel)
                common_artists = set(video_artists) & set(source_artists)
                if common_artists:
                    artists_str = ', '.join(list(common_artists)[:2])
                    reasons.append(f"• Features {artists_str} who you've liked before")
                
                # Check music style/mood
                style_match = detect_music_style(video_title, source_title)
                if style_match:
                    reasons.append(f"• Similar {style_match} style music")
                
                # Add recency if it's a new release
                published_at = item['snippet']['publishedAt']
                if '2024' in published_at:
                    reasons.append("• New release in your preferred style")
                
                # Only add if we have meaningful reasons
                if len(reasons) >= 2:  # Require at least 2 good reasons
                    related_videos.append({
                        'title': video_title,
                        'channel': video_channel,
                        'video_id': item['id']['videoId'],
                        'reason': '\n'.join(reasons)
                    })
        
        logger.info(f"Found {len(related_videos)} content-based recommendations")
        return related_videos[:max_results]
        
    except Exception as e:
        logger.error(f"Error fetching related videos: {str(e)}")
        return []

def extract_artists(title, channel):
    """Extract artist names from video title and channel"""
    # Remove common words and split
    common_words = ['official', 'video', 'music', 'audio', 'lyric', 'ft.', 'feat', 'presents']
    clean_title = ' '.join(word for word in title.lower().split() 
                          if word not in common_words)
    
    # Try to extract artist names
    artists = []
    
    # Check for common patterns
    patterns = [
        r'by\s+([^-|]+)',
        r'ft\.\s+([^-|]+)',
        r'feat\.\s+([^-|]+)',
        r'^([^-|]+)\s*[-|]',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, clean_title)
        artists.extend(matches)
    
    # Add channel name if it looks like an artist
    if not any(word in channel.lower() for word in ['vevo', 'music', 'records', 'official']):
        artists.append(channel)
    
    return list(set(artists))

def detect_music_style(video_title, source_title):
    """Detect music style/mood similarities"""
    styles = {
        'melodic': ['melody', 'melodic', 'tune', 'harmony'],
        'upbeat': ['party', 'dance', 'groove', 'beat', 'rhythm'],
        'romantic': ['love', 'romantic', 'heart', 'soul'],
        'classical fusion': ['classical', 'fusion', 'traditional'],
        'folk inspired': ['folk', 'traditional', 'acoustic'],
    }
    
    for style, keywords in styles.items():
        if any(keyword in video_title.lower() for keyword in keywords) and \
           any(keyword in source_title.lower() for keyword in keywords):
            return style
    
    return None

def generate_similar_videos(videos, count=10, youtube=None):
    """Generate similar video recommendations using YouTube API"""
    logger.info(f"Generating similar videos from {len(videos)} source videos")
    
    if youtube is None:
        # Use the YouTube agent to get API access
        agent = YouTubeAgent()
        youtube = agent.youtube

    # Get sample videos from this category
    sample_videos = videos.sample(min(5, len(videos)))
    recommendations = []
    
    # Get related videos for each sample
    for _, video in sample_videos.iterrows():
        try:
            related = get_related_videos(youtube, video['video_id'])
            recommendations.extend(related)
            
            if len(recommendations) >= count:
                break
        except Exception as e:
            logger.error(f"Error processing video {video['video_id']}: {str(e)}")
            continue
    
    # Remove duplicates and limit to requested count
    seen_ids = set()
    unique_recommendations = []
    for rec in recommendations:
        if rec['video_id'] not in seen_ids:
            seen_ids.add(rec['video_id'])
            unique_recommendations.append(rec)
            if len(unique_recommendations) >= count:
                break
    
    logger.info(f"Generated {len(unique_recommendations)} recommendations")
    return unique_recommendations[:count]

def create_recommendation_row(title, videos, reason_prefix, youtube=None):
    st.subheader(title)
    
    # Generate recommendations for this category
    with st.spinner("Fetching recommendations..."):
        recommendations = generate_similar_videos(videos, youtube=youtube)
    
    if recommendations:
        # Create two rows of 5 videos each
        for i in range(0, len(recommendations), 5):
            cols = st.columns(5)
            for j, video in enumerate(recommendations[i:i+5]):
                with cols[j]:
                    create_youtube_tile(video)
        st.markdown("---")
    else:
        st.info("No recommendations available for this category")

def generate_recommendations(df):
    """Generate music recommendations with YouTube-style video previews"""
    logger.info("Generating personalized recommendations")
    st.header("Recommended For You")
    
    # Initialize YouTube API with proper authentication
    youtube = get_authenticated_youtube()
    if youtube is None:
        st.error("Could not authenticate with YouTube API")
        return
    
    # Get all languages and genres with more than 10 videos
    language_counts = df['language'].value_counts()
    genre_counts = df['genre'].value_counts()
    
    popular_languages = language_counts[language_counts >= 10].index
    popular_genres = genre_counts[genre_counts >= 10].index
    
    logger.info(f"Found {len(popular_languages)} languages and {len(popular_genres)} genres with 10+ videos")

    # Language-based recommendations
    for language in popular_languages:
        language_videos = df[df['language'] == language]
        if len(language_videos) >= 10:
            logger.info(f"Creating recommendation row for {language} with {len(language_videos)} videos")
            create_recommendation_row(
                f"New {language} Music For You",
                language_videos,
                f'New {language} track',
                youtube=youtube
            )

    # Genre-based recommendations
    for genre in popular_genres:
        genre_videos = df[df['genre'] == genre]
        if len(genre_videos) >= 10:
            logger.info(f"Creating recommendation row for {genre} with {len(genre_videos)} videos")
            create_recommendation_row(
                f"New {genre} Recommendations",
                genre_videos,
                f'New in {genre}',
                youtube=youtube
            )

    # Add recommendation summary
    st.info(f"""Showing new recommendations based on:
    - Your favorite languages: {', '.join(popular_languages)}
    - Your favorite genres: {', '.join(popular_genres)}
    - Using YouTube's recommendation engine
    """)

def cache_llm_response(video_title, detected_language, reason):
    """Cache LLM detection results"""
    cache_file = "llm_cache.json"
    try:
        # Load existing cache
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        else:
            cache = {}
        
        # Create hash of the title as key
        title_hash = hashlib.md5(video_title.encode()).hexdigest()
        
        # Store the result
        cache[title_hash] = {
            'title': video_title,
            'language': detected_language,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save updated cache
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Cached LLM result for: {video_title[:50]}...")
        return True
    except Exception as e:
        logger.error(f"Error caching LLM result: {str(e)}")
        return False

def get_cached_llm_response(video_title):
    """Get cached LLM detection result if available"""
    cache_file = "llm_cache.json"
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            
            # Get result using title hash
            title_hash = hashlib.md5(video_title.encode()).hexdigest()
            if title_hash in cache:
                result = cache[title_hash]
                logger.info(f"Found cached LLM result for: {video_title[:50]}...")
                return result['language'], result['reason']
        return None, None
    except Exception as e:
        logger.error(f"Error reading LLM cache: {str(e)}")
        return None, None

def main():
    try:
        logger.info("Initializing application...")
        st.set_page_config(page_title="YouTube Liked Videos Organizer", layout="wide")
        
        # Initialize session state
        if 'filter_type' not in st.session_state:
            st.session_state['filter_type'] = None
        if 'filter_value' not in st.session_state:
            st.session_state['filter_value'] = None
        
        st.title("YouTube Liked Videos Organizer")

        # Load cache data silently
        logger.info("Checking for cached data...")
        cache_data = load_from_cache()
        if cache_data:
            logger.info(f"Found cached data with {len(cache_data['music_videos'])} music videos")
            music_videos = cache_data['music_videos']
            
            # If cache exists, directly show analytics
            if music_videos:
                df_music = analyze_videos(music_videos)
                display_analytics(df_music)
            
            # Add refresh button
            if st.button("Refresh Data", type="secondary"):
                if clear_cache():
                    st.success("Cache cleared, fetching fresh data...")
                    st.rerun()
        else:
            # Show load button if no cache exists
            logger.info("No cached data found")
            st.info("Click 'Load Videos' to start")
            if st.button("Load Videos", type="primary"):
                try:
                    with st.spinner("Connecting to YouTube..."):
                        agent = YouTubeAgent()
                        agent.authenticate()
                        
                        liked_videos = agent.get_liked_videos()
                        music_videos, other_videos = agent.categorize_videos(liked_videos)
                        
                        # Save to cache
                        cache_saved = save_to_cache({
                            'liked_videos': liked_videos,
                            'music_videos': music_videos,
                            'other_videos': other_videos
                        })
                        if cache_saved:
                            st.success("Data loaded successfully!")
                            st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error("Error in video processing", exc_info=True)
                    return

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error("Application error", exc_info=True)

if __name__ == "__main__":
    main() 