from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import os
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class YouTubeAgent:
    def __init__(self):
        self.api_service_name = "youtube"
        self.api_version = "v3"
        self.client_secrets_file = "client_secrets.json"
        self.scopes = ['https://www.googleapis.com/auth/youtube.readonly']
        self.credentials = None
        self.youtube = None

    def authenticate(self):
        """Handles OAuth2 authentication with YouTube."""
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                self.credentials = pickle.load(token)

        if not self.credentials or not self.credentials.valid:
            try:
                if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                    self.credentials.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.client_secrets_file, self.scopes)
                    self.credentials = flow.run_local_server(
                        port=0,
                        success_message='Authentication successful! You can close this window.',
                        open_browser=True
                    )

                with open('token.pickle', 'wb') as token:
                    pickle.dump(self.credentials, token)

            except Exception as e:
                raise Exception(
                    f"Authentication failed: {str(e)}\n\n"
                    "To fix this:\n"
                    "1. Go to Google Cloud Console (https://console.cloud.google.com)\n"
                    "2. Select your project\n"
                    "3. Go to APIs & Services > OAuth consent screen\n"
                    "4. Set up the OAuth consent screen if not done already\n"
                    "5. Add your Google account email as a test user\n"
                    "6. Go to APIs & Services > Credentials\n"
                    "7. Create or update your OAuth 2.0 Client ID\n"
                    "8. Download the client secrets file and save as 'client_secrets.json'\n"
                )

        self.youtube = build(
            self.api_service_name, 
            self.api_version, 
            credentials=self.credentials
        )

    def get_liked_videos(self):
        """Get all liked videos with descriptions"""
        try:
            videos = []
            next_page_token = None
            
            while True:
                request = self.youtube.videos().list(
                    part="snippet,contentDetails",
                    myRating="like",
                    maxResults=50,
                    pageToken=next_page_token
                )
                response = request.execute()
                
                for item in response['items']:
                    video = {
                        'video_id': item['id'],
                        'title': item['snippet']['title'],
                        'description': item['snippet']['description'],  # Include description
                        'channel_title': item['snippet']['channelTitle'],
                        'published_at': item['snippet']['publishedAt'],
                    }
                    videos.append(video)
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
            return videos
        except Exception as e:
            logger.error(f"Error fetching liked videos: {e}")
            return []

    def categorize_videos(self, videos):
        """Categorize videos into music and non-music"""
        music_videos = []
        other_videos = []
        
        for video in videos:
            try:
                # Get video details to check category
                video_request = self.youtube.videos().list(
                    part="snippet",
                    id=video['video_id']
                ).execute()
                
                if video_request.get('items'):
                    # Get category ID from snippet
                    category_id = video_request['items'][0]['snippet'].get('categoryId')
                    
                    # Category 10 is Music in YouTube's category system
                    if category_id == '10':
                        music_videos.append(video)
                    else:
                        # Add category info to video dict
                        video['category_id'] = category_id
                        other_videos.append(video)
                else:
                    # If can't get category, default to other
                    other_videos.append(video)
                    
            except Exception as e:
                logger.error(f"Error categorizing video {video.get('video_id', 'unknown')}: {str(e)}")
                # If error occurs, put in other videos
                other_videos.append(video)
        
        logger.info(f"Categorized {len(music_videos)} music videos and {len(other_videos)} other videos")
        return music_videos, other_videos

    def export_to_csv(self, music_videos, other_videos):
        """Exports the categorized videos to CSV files."""
        music_df = pd.DataFrame(music_videos)
        other_df = pd.DataFrame(other_videos)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        music_df.to_csv(f'music_videos_{timestamp}.csv', index=False)
        other_df.to_csv(f'other_videos_{timestamp}.csv', index=False)

        return f'music_videos_{timestamp}.csv', f'other_videos_{timestamp}.csv' 