import streamlit as st
import logging

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting test app")
    
    try:
        st.title("Test App")
        st.write("If you can see this, Streamlit is working!")
        
        if st.button("Click me"):
            st.write("Button clicked!")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 