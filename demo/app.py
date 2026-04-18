"""Streamlit demo application for face authentication system."""

import io
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import torch
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Face Authentication System",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticator' not in st.session_state:
    st.session_state.authenticator = None
if 'enrolled_users' not in st.session_state:
    st.session_state.enrolled_users = []


def load_authenticator() -> Optional[object]:
    """Load the face authenticator."""
    try:
        # Import here to avoid issues if dependencies are missing
        from src.models.face_authentication import FaceAuthenticator
        
        # Initialize authenticator with default config
        authenticator = FaceAuthenticator()
        return authenticator
    except Exception as e:
        st.error(f"Failed to load authenticator: {str(e)}")
        return None


def display_disclaimer():
    """Display disclaimer and privacy notice."""
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ DISCLAIMER</h4>
        <p><strong>This is a defensive research demonstration for educational purposes only.</strong></p>
        <ul>
            <li>Not intended for production security operations</li>
            <li>May contain inaccuracies and should not be used as a SOC tool</li>
            <li>Designed for research and educational use in controlled environments</li>
            <li>No exploitation or offensive capabilities included</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def display_privacy_notice():
    """Display privacy notice."""
    st.markdown("""
    <div class="success-box">
        <h4>🔒 Privacy Protection</h4>
        <ul>
            <li>All processing is performed locally on your device</li>
            <li>No images or biometric data are transmitted to external servers</li>
            <li>All sample data is synthetic and de-identified</li>
            <li>User consent is required for all biometric enrollment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def process_uploaded_image(uploaded_file) -> Optional[Image.Image]:
    """Process uploaded image file."""
    try:
        # Convert uploaded file to PIL Image
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None


def display_authentication_result(result):
    """Display authentication result."""
    if result.is_authenticated:
        st.markdown(f"""
        <div class="success-box">
            <h4>✅ Authentication Successful</h4>
            <p><strong>User:</strong> {result.user_name}</p>
            <p><strong>Confidence:</strong> {result.confidence:.3f}</p>
            <p><strong>Explanation:</strong> {result.explanation}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="error-box">
            <h4>❌ Authentication Failed</h4>
            <p><strong>Confidence:</strong> {result.confidence:.3f}</p>
            <p><strong>Explanation:</strong> {result.explanation}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display additional metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Liveness Score", f"{result.liveness_score:.3f}" if result.liveness_score else "N/A")
    
    with col2:
        st.metric("Anti-Spoofing Score", f"{result.anti_spoofing_score:.3f}" if result.anti_spoofing_score else "N/A")
    
    with col3:
        st.metric("Quality Score", f"{result.quality_score:.3f}" if result.quality_score else "N/A")


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🔐 Face Authentication System</h1>', unsafe_allow_html=True)
    
    # Display disclaimer and privacy notice
    display_disclaimer()
    display_privacy_notice()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "User Enrollment", "Authentication", "System Status", "Evaluation"]
    )
    
    # Load authenticator
    if st.session_state.authenticator is None:
        with st.spinner("Loading face authentication system..."):
            st.session_state.authenticator = load_authenticator()
    
    if st.session_state.authenticator is None:
        st.error("Failed to load face authentication system. Please check the installation.")
        return
    
    # Main content based on selected page
    if page == "Home":
        show_home_page()
    elif page == "User Enrollment":
        show_enrollment_page()
    elif page == "Authentication":
        show_authentication_page()
    elif page == "System Status":
        show_status_page()
    elif page == "Evaluation":
        show_evaluation_page()


def show_home_page():
    """Display home page."""
    st.header("Welcome to the Face Authentication System")
    
    st.markdown("""
    This system demonstrates modern face authentication with the following features:
    
    ### 🔍 Core Features
    - **Face Recognition**: Advanced FaceNet/ArcFace-based face recognition
    - **Liveness Detection**: Anti-spoofing measures to prevent photo/video attacks
    - **Privacy Protection**: Local processing with no cloud dependencies
    - **Comprehensive Evaluation**: EER, minDCF, ROC/DET curves, FAR/FRR metrics
    
    ### 🛡️ Security Features
    - **Anti-Spoofing**: Multiple detection mechanisms for presentation attacks
    - **Liveness Detection**: Prevents photo/video spoofing attacks
    - **Threshold Management**: Configurable security vs usability trade-offs
    - **Audit Logging**: Comprehensive logging of authentication attempts
    
    ### 📊 Evaluation Metrics
    - **EER (Equal Error Rate)**: Primary biometric performance metric
    - **minDCF (Minimum Detection Cost Function)**: Cost-sensitive evaluation
    - **ROC/DET Curves**: Visualization of authentication performance
    - **FAR/FRR**: False Acceptance and False Rejection Rates
    
    ### 🚀 Getting Started
    1. **Enroll Users**: Go to the "User Enrollment" page to add users to the system
    2. **Authenticate**: Use the "Authentication" page to verify user identity
    3. **Monitor**: Check "System Status" to view enrolled users and system metrics
    4. **Evaluate**: Run evaluations to assess system performance
    """)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Enrolled Users", len(st.session_state.authenticator.enrolled_users))
    
    with col2:
        st.metric("System Status", "Active")
    
    with col3:
        st.metric("Model Type", "FaceNet + Liveness Detection")
    
    with col4:
        st.metric("Privacy Level", "Local Processing Only")


def show_enrollment_page():
    """Display user enrollment page."""
    st.header("User Enrollment")
    
    st.markdown("""
    Enroll new users in the face authentication system. Each user needs to provide a clear face image for enrollment.
    """)
    
    # User information form
    with st.form("enrollment_form"):
        st.subheader("User Information")
        
        user_id = st.text_input("User ID", placeholder="Enter unique user identifier")
        user_name = st.text_input("User Name (Optional)", placeholder="Enter user's display name")
        
        st.subheader("Enrollment Image")
        uploaded_file = st.file_uploader(
            "Upload face image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear face image for enrollment"
        )
        
        submitted = st.form_submit_button("Enroll User")
        
        if submitted:
            if not user_id:
                st.error("Please enter a User ID")
            elif not uploaded_file:
                st.error("Please upload an image")
            else:
                # Process enrollment
                with st.spinner("Processing enrollment..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Enroll user
                        success = st.session_state.authenticator.enroll_user(
                            user_id=user_id,
                            image_path=tmp_path,
                            user_name=user_name or user_id
                        )
                        
                        if success:
                            st.success(f"User {user_id} enrolled successfully!")
                            
                            # Display enrolled image
                            image = Image.open(tmp_path)
                            st.image(image, caption=f"Enrolled image for {user_id}", width=300)
                        else:
                            st.error("Failed to enroll user. Please check the image and try again.")
                    
                    except Exception as e:
                        st.error(f"Enrollment error: {str(e)}")
                    
                    finally:
                        # Clean up temporary file
                        Path(tmp_path).unlink(missing_ok=True)


def show_authentication_page():
    """Display authentication page."""
    st.header("Face Authentication")
    
    st.markdown("""
    Authenticate a user by uploading their face image. The system will compare the image against enrolled users.
    """)
    
    # Check if users are enrolled
    if not st.session_state.authenticator.enrolled_users:
        st.warning("No users enrolled in the system. Please enroll users first.")
        return
    
    # Authentication form
    with st.form("authentication_form"):
        st.subheader("Authentication")
        
        uploaded_file = st.file_uploader(
            "Upload face image for authentication",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a face image to authenticate"
        )
        
        # Optional: select specific user
        user_options = ["All Users"] + list(st.session_state.authenticator.enrolled_users.keys())
        selected_user = st.selectbox("Authenticate against", user_options)
        
        # Threshold adjustment
        threshold = st.slider(
            "Authentication Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Higher values = more strict authentication"
        )
        
        submitted = st.form_submit_button("Authenticate")
        
        if submitted:
            if not uploaded_file:
                st.error("Please upload an image")
            else:
                # Process authentication
                with st.spinner("Processing authentication..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Authenticate
                        user_id = None if selected_user == "All Users" else selected_user
                        result = st.session_state.authenticator.authenticate(
                            image_path=tmp_path,
                            user_id=user_id,
                            threshold=threshold
                        )
                        
                        # Display result
                        display_authentication_result(result)
                        
                        # Display uploaded image
                        image = Image.open(tmp_path)
                        st.image(image, caption="Authentication image", width=300)
                    
                    except Exception as e:
                        st.error(f"Authentication error: {str(e)}")
                    
                    finally:
                        # Clean up temporary file
                        Path(tmp_path).unlink(missing_ok=True)


def show_status_page():
    """Display system status page."""
    st.header("System Status")
    
    # System metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Enrolled Users", len(st.session_state.authenticator.enrolled_users))
    
    with col2:
        st.metric("System Status", "Active")
    
    with col3:
        st.metric("Model Device", str(st.session_state.authenticator.device))
    
    # Enrolled users table
    st.subheader("Enrolled Users")
    
    if st.session_state.authenticator.enrolled_users:
        users_data = []
        for user_id, user_data in st.session_state.authenticator.enrolled_users.items():
            users_data.append({
                'User ID': user_id,
                'User Name': user_data['user_name'],
                'Enrollment Image': user_data['enrollment_image']
            })
        
        st.dataframe(users_data, use_container_width=True)
        
        # User management
        st.subheader("User Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Refresh User List"):
                st.rerun()
        
        with col2:
            user_to_remove = st.selectbox(
                "Select user to remove",
                ["Select user..."] + list(st.session_state.authenticator.enrolled_users.keys())
            )
            
            if st.button("Remove User") and user_to_remove != "Select user...":
                if st.session_state.authenticator.remove_user(user_to_remove):
                    st.success(f"User {user_to_remove} removed successfully")
                    st.rerun()
                else:
                    st.error(f"Failed to remove user {user_to_remove}")
    else:
        st.info("No users enrolled in the system.")


def show_evaluation_page():
    """Display evaluation page."""
    st.header("System Evaluation")
    
    st.markdown("""
    Evaluate the face authentication system performance using test data.
    """)
    
    # Evaluation controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Evaluation Settings")
        
        p_target = st.slider(
            "Target Probability (p_target)",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.3f"
        )
        
        c_miss = st.slider(
            "Cost of Miss (c_miss)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1
        )
        
        c_fa = st.slider(
            "Cost of False Alarm (c_fa)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1
        )
    
    with col2:
        st.subheader("Test Data")
        
        test_data_dir = st.text_input(
            "Test Data Directory",
            value="data/test",
            help="Directory containing test data"
        )
        
        if st.button("Run Evaluation"):
            if not Path(test_data_dir).exists():
                st.error(f"Test data directory not found: {test_data_dir}")
            else:
                with st.spinner("Running evaluation..."):
                    try:
                        # Run evaluation
                        results = st.session_state.authenticator.evaluate_system(
                            test_data_dir=test_data_dir,
                            output_dir="assets/evaluation"
                        )
                        
                        # Display results
                        st.success("Evaluation completed successfully!")
                        
                        # Authentication results
                        st.subheader("Authentication Results")
                        auth_results = results['authentication']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("EER", f"{auth_results['eer']:.4f}")
                        
                        with col2:
                            st.metric("MinDCF", f"{auth_results['min_dcf']:.4f}")
                        
                        with col3:
                            st.metric("ROC AUC", f"{auth_results['roc_auc']:.4f}")
                        
                        with col4:
                            st.metric("PR AUC", f"{auth_results['pr_auc']:.4f}")
                        
                        # Liveness detection results
                        st.subheader("Liveness Detection Results")
                        liveness_results = results['liveness_detection']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Accuracy", f"{liveness_results['accuracy']:.4f}")
                        
                        with col2:
                            st.metric("Precision", f"{liveness_results['precision']:.4f}")
                        
                        with col3:
                            st.metric("Recall", f"{liveness_results['recall']:.4f}")
                        
                        with col4:
                            st.metric("F1 Score", f"{liveness_results['f1_score']:.4f}")
                        
                        # Display plots if available
                        plots_dir = Path("assets/evaluation")
                        if plots_dir.exists():
                            st.subheader("Evaluation Plots")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if (plots_dir / "roc_curve.png").exists():
                                    st.image(str(plots_dir / "roc_curve.png"), caption="ROC Curve")
                            
                            with col2:
                                if (plots_dir / "det_curve.png").exists():
                                    st.image(str(plots_dir / "det_curve.png"), caption="DET Curve")
                            
                            if (plots_dir / "score_distributions.png").exists():
                                st.image(str(plots_dir / "score_distributions.png"), caption="Score Distributions")
                    
                    except Exception as e:
                        st.error(f"Evaluation failed: {str(e)}")


if __name__ == "__main__":
    main()
