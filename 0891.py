#!/usr/bin/env python3
"""
Project 891: Modern Face Authentication System

This is a modernized, research-focused face authentication system with:
- Advanced FaceNet/ArcFace-based face recognition
- Liveness detection and anti-spoofing measures
- Comprehensive biometric evaluation metrics
- Privacy-first design with local processing
- Interactive Streamlit demo application

This file demonstrates the basic usage of the modernized system.
For the full implementation, see the src/ directory and demo application.
"""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Demonstrate basic usage of the face authentication system."""
    try:
        # Import the modern face authentication system
        from src.models.face_authentication import FaceAuthenticator
        
        logger.info("Initializing Face Authentication System...")
        
        # Initialize the authenticator
        authenticator = FaceAuthenticator()
        
        logger.info("Face Authentication System initialized successfully!")
        logger.info(f"Using device: {authenticator.device}")
        
        # Display system information
        print("\n" + "="*60)
        print("🔐 FACE AUTHENTICATION SYSTEM")
        print("="*60)
        print(f"Model Type: FaceNet + Liveness Detection")
        print(f"Device: {authenticator.device}")
        print(f"Privacy Level: Local Processing Only")
        print("="*60)
        
        print("\n📋 Available Features:")
        print("• Advanced face recognition with FaceNet/ArcFace")
        print("• Liveness detection to prevent spoofing attacks")
        print("• Anti-spoofing measures for presentation attacks")
        print("• Comprehensive biometric evaluation metrics")
        print("• Privacy-first design with local processing")
        print("• Interactive Streamlit demo application")
        
        print("\n🚀 Quick Start:")
        print("1. Run the demo: streamlit run demo/app.py")
        print("2. Train a model: python scripts/train.py")
        print("3. Evaluate system: python scripts/evaluate.py")
        
        print("\n⚠️  DISCLAIMER:")
        print("This is a defensive research demonstration for educational purposes only.")
        print("Not intended for production security operations.")
        
        logger.info("Demo completed successfully!")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print("\n❌ Error: Required dependencies not installed.")
        print("Please install dependencies: pip install -e .")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()