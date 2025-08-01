#!/bin/bash
# Production deployment script for Research Paper System

echo "🚀 Starting Research Paper System deployment..."

# Update system packages
echo "📦 Updating system packages..."
apt-get update

# Install FFmpeg (required for audio processing)
echo "🎵 Installing FFmpeg..."
apt-get install -y ffmpeg

# Verify FFmpeg installation
if command -v ffmpeg >/dev/null 2>&1; then
    echo "✅ FFmpeg installed successfully"
else
    echo "❌ FFmpeg installation failed - audio merging will not work"
fi

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -r requirements.txt

# Verify critical dependencies
echo "🔍 Verifying dependencies..."
python -c "import streamlit, gtts, pydub; print('✅ Core dependencies installed')" 2>/dev/null || echo "❌ Some dependencies missing"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/audio data/uploads data/state

# Set permissions
chmod 755 data/audio data/uploads data/state

echo "🎉 Deployment setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Set environment variables (GEMINI_API_KEY or OPENAI_API_KEY)"
echo "2. Start the application: streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0"
echo "3. Verify audio generation works by testing with a few papers"
