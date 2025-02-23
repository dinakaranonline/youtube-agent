from youtube_agent import YouTubeAgent

def main():
    # Initialize the YouTube agent
    agent = YouTubeAgent()
    
    try:
        # Authenticate with YouTube
        print("Authenticating with YouTube...")
        agent.authenticate()

        # Fetch liked videos
        print("Fetching liked videos...")
        liked_videos = agent.get_liked_videos()
        print(f"Found {len(liked_videos)} liked videos")

        # Categorize videos
        print("Categorizing videos...")
        music_videos, other_videos = agent.categorize_videos(liked_videos)
        print(f"Found {len(music_videos)} music videos and {len(other_videos)} other videos")

        # Export to CSV
        print("Exporting to CSV...")
        music_file, other_file = agent.export_to_csv(music_videos, other_videos)
        print(f"Exported music videos to: {music_file}")
        print(f"Exported other videos to: {other_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 