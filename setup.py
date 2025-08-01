#!/usr/bin/env python3
"""
Cross-platform setup script for Research Paper System.
Installs FFmpeg and Python dependencies.
Supports Windows, macOS, and Linux distributions.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path


def is_ffmpeg_installed():
    """Check if FFmpeg is already installed and accessible."""
    return shutil.which("ffmpeg") is not None


def install_python_requirements():
    """Install Python requirements from requirements.txt."""
    print("\n📦 Installing Python requirements...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found in the same directory as this script")
        return False
    
    try:
        # Use pip to install requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Python requirements installed successfully")
            return True
        else:
            print(f"❌ Failed to install requirements: {result.stderr}")
            
            # Check for externally-managed environment error
            if "externally-managed-environment" in result.stderr:
                print("\n💡 Detected externally-managed Python environment.")
                print("   You have two options:")
                print("   1. Create a virtual environment:")
                print("      python -m venv research_env")
                print("      research_env\\Scripts\\activate  # Windows")
                print("      source research_env/bin/activate  # Linux/macOS")
                print("      python -m pip install -r requirements.txt")
                print("   2. Use --break-system-packages flag (not recommended):")
                print("      python -m pip install -r requirements.txt --break-system-packages")
                
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Requirements installation timed out")
        return False
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        return False


def install_ffmpeg_windows():
    """Install FFmpeg on Windows using winget."""
    print("Installing FFmpeg on Windows...")
    
    # Try winget first (Windows 10/11)
    try:
        result = subprocess.run(
            ["winget", "install", "Gyan.FFmpeg"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print("✅ FFmpeg installed successfully via winget")
            
            # Add to PATH
            ffmpeg_path = f"C:\\Users\\{os.environ.get('USERNAME')}\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-7.1.1-full_build\\bin"
            
            # Add to user PATH
            try:
                current_path = subprocess.check_output(
                    ['reg', 'query', 'HKCU\\Environment', '/v', 'PATH'],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                
                if ffmpeg_path not in current_path:
                    subprocess.run(['setx', 'PATH', f"{current_path.split()[-1]};{ffmpeg_path}"])
                    print("✅ FFmpeg added to PATH")
                else:
                    print("✅ FFmpeg already in PATH")
                    
            except subprocess.CalledProcessError:
                # PATH doesn't exist, create it
                subprocess.run(['setx', 'PATH', ffmpeg_path])
                print("✅ FFmpeg PATH created")
            
            # Update current environment PATH for immediate availability
            os.environ["PATH"] = f"{os.environ.get('PATH', '')};{ffmpeg_path}"
            
            # Verify installation by checking if ffmpeg is now accessible
            if shutil.which("ffmpeg"):
                print("✅ FFmpeg is now accessible in current session")
                return True
            else:
                print("⚠️  FFmpeg installed but requires terminal restart to be accessible")
                return True  # Still consider it successful since it's installed
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ winget not available or failed")
    
    # Fallback: Manual instructions
    print("\n📝 Manual Installation Required:")
    print("1. Download FFmpeg from: https://www.gyan.dev/ffmpeg/builds/")
    print("2. Extract to C:\\ffmpeg\\")
    print("3. Add C:\\ffmpeg\\bin to your PATH environment variable")
    print("4. Restart your terminal/VS Code")
    
    return False


def install_ffmpeg_macos():
    """Install FFmpeg on macOS using Homebrew."""
    print("Installing FFmpeg on macOS...")
    
    # Check if Homebrew is installed
    if not shutil.which("brew"):
        print("❌ Homebrew not found. Please install Homebrew first:")
        print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        return False
    
    try:
        result = subprocess.run(
            ["brew", "install", "ffmpeg"],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode == 0:
            print("✅ FFmpeg installed successfully via Homebrew")
            return True
        else:
            print(f"❌ Homebrew installation failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Installation timed out")
        return False


def install_ffmpeg_linux():
    """Install FFmpeg on Linux using package managers."""
    print("Installing FFmpeg on Linux...")
    
    # Detect Linux distribution
    try:
        with open("/etc/os-release") as f:
            os_info = f.read().lower()
    except FileNotFoundError:
        os_info = ""
    
    # Try different package managers
    package_managers = [
        # Ubuntu/Debian
        (["apt", "update"], ["apt", "install", "-y", "ffmpeg"]),
        # CentOS/RHEL/Fedora
        (["yum", "update", "-y"], ["yum", "install", "-y", "ffmpeg"]),
        # Fedora (newer)
        (["dnf", "update", "-y"], ["dnf", "install", "-y", "ffmpeg"]),
        # Arch Linux
        (None, ["pacman", "-S", "--noconfirm", "ffmpeg"]),
        # Alpine Linux
        (["apk", "update"], ["apk", "add", "ffmpeg"]),
    ]
    
    for update_cmd, install_cmd in package_managers:
        if shutil.which(install_cmd[0]):
            try:
                print(f"Using {install_cmd[0]} package manager...")
                
                # Update package list if needed
                if update_cmd:
                    subprocess.run(update_cmd, check=True, timeout=300)
                
                # Install FFmpeg
                result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0:
                    print("✅ FFmpeg installed successfully")
                    return True
                else:
                    print(f"❌ Installation failed with {install_cmd[0]}: {result.stderr}")
                    
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"❌ Error with {install_cmd[0]}: {e}")
                continue
    
    # Fallback: Manual instructions
    print("\n📝 Manual Installation Required:")
    print("Please install FFmpeg using your distribution's package manager:")
    print("  Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg")
    print("  CentOS/RHEL:   sudo yum install ffmpeg")
    print("  Fedora:        sudo dnf install ffmpeg")
    print("  Arch Linux:    sudo pacman -S ffmpeg")
    print("  Alpine:        sudo apk add ffmpeg")
    
    return False


def main():
    """Main setup function."""
    print("🚀 Complete Setup for Research Paper System")
    print("=" * 50)
    
    # Step 1: Install Python requirements
    print("Step 1: Installing Python dependencies...")
    requirements_success = install_python_requirements()
    
    # Step 2: Check and install FFmpeg
    print("\nStep 2: Setting up FFmpeg...")
    if is_ffmpeg_installed():
        print("✅ FFmpeg is already installed and accessible!")
        print(f"   Location: {shutil.which('ffmpeg')}")
        ffmpeg_success = True
    else:
        print("❌ FFmpeg not found in PATH. Installing...")
        
        # Detect operating system
        system = platform.system().lower()
        
        if system == "windows":
            ffmpeg_success = install_ffmpeg_windows()
        elif system == "darwin":  # macOS
            ffmpeg_success = install_ffmpeg_macos()
        elif system == "linux":
            ffmpeg_success = install_ffmpeg_linux()
        else:
            print(f"❌ Unsupported operating system: {system}")
            ffmpeg_success = False
        
        # Re-check after installation
        if ffmpeg_success and not is_ffmpeg_installed():
            print("⚠️  FFmpeg installed successfully but not yet accessible in current session")
            print("   This is normal - PATH changes require a new terminal session")
            # Still consider it successful since FFmpeg was installed
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Setup Summary:")
    print(f"   Python Requirements: {'✅ Success' if requirements_success else '❌ Failed'}")
    print(f"   FFmpeg Installation: {'✅ Success' if ffmpeg_success else '❌ Failed'}")
    
    if requirements_success and ffmpeg_success:
        print("\n🎉 Complete setup successful!")
        print("📝 Next steps:")
        if not is_ffmpeg_installed():
            print("   1. ⚠️  IMPORTANT: Restart your terminal or VS Code for FFmpeg to work")
            print("   2. Run: streamlit run app.py")
            print("   3. All features including audio generation will work properly")
        else:
            print("   1. Run: streamlit run app.py")
            print("   2. All features including audio generation will work properly")
    elif requirements_success:
        print("\n⚠️  Python requirements installed, but FFmpeg setup failed.")
        print("   The app will work, but audio features may not function.")
        print("   Please install FFmpeg manually and ensure it's in your PATH.")
    elif ffmpeg_success:
        print("\n⚠️  FFmpeg installed, but Python requirements failed.")
        print("   Please install Python dependencies manually:")
        print("   python -m pip install -r requirements.txt")
        if not is_ffmpeg_installed():
            print("   Also restart your terminal for FFmpeg to work properly.")
    else:
        print("\n❌ Setup incomplete. Please resolve the issues above.")
    
    return requirements_success and ffmpeg_success


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
