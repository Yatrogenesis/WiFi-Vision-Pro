#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Vision Pro - Cross-Platform Installer Builder
Creates installers for Windows, Linux, and macOS
"""

import os
import sys
import shutil
import subprocess
import platform
import json
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime

class CrossPlatformBuilder:
    """Build installers for multiple platforms"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.build_dir = self.project_dir / "build"
        self.dist_dir = self.project_dir / "dist"
        self.version = "2.0.0"
        self.app_name = "WiFi Vision Pro"
        
        # Create build directories
        self.build_dir.mkdir(exist_ok=True)
        self.dist_dir.mkdir(exist_ok=True)
        
        # Platform-specific settings
        self.current_platform = platform.system().lower()
        
    def build_for_current_platform(self):
        """Build installer for current platform"""
        print(f"Building WiFi Vision Pro v{self.version} for {self.current_platform}")
        print("=" * 60)
        
        if self.current_platform == "windows":
            return self.build_windows()
        elif self.current_platform == "linux":
            return self.build_linux()
        elif self.current_platform == "darwin":
            return self.build_macos()
        else:
            print(f"Unsupported platform: {self.current_platform}")
            return False
    
    def build_windows(self) -> bool:
        """Build Windows installer"""
        print("Building Windows installer...")
        
        try:
            # Step 1: Create PyInstaller spec file
            spec_content = self.create_pyinstaller_spec_windows()
            spec_file = self.build_dir / "wifi_vision_pro_windows.spec"
            with open(spec_file, 'w') as f:
                f.write(spec_content)
            
            # Step 2: Build with PyInstaller
            print("Running PyInstaller...")
            cmd = [
                sys.executable, "-m", "PyInstaller",
                "--clean",
                "--noconfirm",
                str(spec_file)
            ]
            
            result = subprocess.run(cmd, cwd=self.project_dir, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"PyInstaller failed: {result.stderr}")
                return False
            
            # Step 3: Create NSIS installer script
            nsis_script = self.create_nsis_script()
            nsis_file = self.build_dir / "wifi_vision_pro_installer.nsi"
            with open(nsis_file, 'w', encoding='utf-8') as f:
                f.write(nsis_script)
            
            # Step 4: Build NSIS installer
            print("Creating NSIS installer...")
            try:
                nsis_cmd = ["makensis", str(nsis_file)]
                subprocess.run(nsis_cmd, check=True, cwd=self.project_dir)
                print("Windows installer created successfully!")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("NSIS not found, creating portable package instead...")
                return self.create_windows_portable()
                
        except Exception as e:
            print(f"Windows build failed: {e}")
            return False
    
    def build_linux(self) -> bool:
        """Build Linux installer"""
        print("Building Linux installer...")
        
        try:
            # Step 1: Create PyInstaller spec file
            spec_content = self.create_pyinstaller_spec_linux()
            spec_file = self.build_dir / "wifi_vision_pro_linux.spec"
            with open(spec_file, 'w') as f:
                f.write(spec_content)
            
            # Step 2: Build with PyInstaller
            print("Running PyInstaller...")
            cmd = [
                sys.executable, "-m", "PyInstaller",
                "--clean",
                "--noconfirm",
                str(spec_file)
            ]
            
            result = subprocess.run(cmd, cwd=self.project_dir, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"PyInstaller failed: {result.stderr}")
                return False
            
            # Step 3: Create AppImage or DEB package
            print("Creating Linux package...")
            self.create_linux_appimage()
            self.create_linux_deb()
            
            print("Linux packages created successfully!")
            return True
            
        except Exception as e:
            print(f"Linux build failed: {e}")
            return False
    
    def build_macos(self) -> bool:
        """Build macOS installer"""
        print("Building macOS installer...")
        
        try:
            # Step 1: Create PyInstaller spec file
            spec_content = self.create_pyinstaller_spec_macos()
            spec_file = self.build_dir / "wifi_vision_pro_macos.spec"
            with open(spec_file, 'w') as f:
                f.write(spec_content)
            
            # Step 2: Build with PyInstaller
            print("Running PyInstaller...")
            cmd = [
                sys.executable, "-m", "PyInstaller",
                "--clean",
                "--noconfirm",
                str(spec_file)
            ]
            
            result = subprocess.run(cmd, cwd=self.project_dir, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"PyInstaller failed: {result.stderr}")
                return False
            
            # Step 3: Create DMG installer
            print("Creating DMG installer...")
            self.create_macos_dmg()
            
            print("macOS installer created successfully!")
            return True
            
        except Exception as e:
            print(f"macOS build failed: {e}")
            return False
    
    def create_pyinstaller_spec_windows(self) -> str:
        """Create PyInstaller spec file for Windows"""
        project_dir_str = str(self.project_dir).replace('\\', '/')
        return f'''# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

block_cipher = None

# Main application
a = Analysis(
    ['advanced_gui.py'],
    pathex=[r'{project_dir_str}'],
    binaries=[],
    datas=[
        ('*.py', '.'),
        ('README.md', '.'),
        ('LICENSE', '.'),
    ],
    hiddenimports=[
        'PySide6.QtCore',
        'PySide6.QtGui', 
        'PySide6.QtWidgets',
        'PySide6.QtMultimedia',
        'PySide6.QtCharts',
        'torch',
        'transformers',
        'diffusers',
        'librosa',
        'soundfile',
        'cv2',
        'numpy',
        'scapy.all',
        'psutil',
        'matplotlib',
        'seaborn'
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=['tkinter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='WiFiVisionPro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if Path('icon.ico').exists() else None,
    version_file='version_info.txt' if Path('version_info.txt').exists() else None,
)
'''
    
    def create_pyinstaller_spec_linux(self) -> str:
        """Create PyInstaller spec file for Linux"""
        return f'''# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

block_cipher = None

# Main application
a = Analysis(
    ['advanced_gui.py'],
    pathex=['{self.project_dir}'],
    binaries=[],
    datas=[
        ('*.py', '.'),
        ('README.md', '.'),
        ('LICENSE', '.'),
    ],
    hiddenimports=[
        'PySide6.QtCore',
        'PySide6.QtGui', 
        'PySide6.QtWidgets',
        'PySide6.QtMultimedia',
        'PySide6.QtCharts',
        'torch',
        'transformers',
        'diffusers',
        'librosa',
        'soundfile',
        'cv2',
        'numpy',
        'scapy.all',
        'psutil',
        'matplotlib',
        'seaborn'
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=['tkinter'],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='WiFiVisionPro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''
    
    def create_pyinstaller_spec_macos(self) -> str:
        """Create PyInstaller spec file for macOS"""
        return f'''# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

block_cipher = None

# Main application
a = Analysis(
    ['advanced_gui.py'],
    pathex=['{self.project_dir}'],
    binaries=[],
    datas=[
        ('*.py', '.'),
        ('README.md', '.'),
        ('LICENSE', '.'),
    ],
    hiddenimports=[
        'PySide6.QtCore',
        'PySide6.QtGui', 
        'PySide6.QtWidgets',
        'PySide6.QtMultimedia',
        'PySide6.QtCharts',
        'torch',
        'transformers',
        'diffusers',
        'librosa',
        'soundfile',
        'cv2',
        'numpy',
        'scapy.all',
        'psutil',
        'matplotlib',
        'seaborn'
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=['tkinter'],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WiFiVisionPro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='WiFiVisionPro',
)

app = BUNDLE(
    coll,
    name='WiFi Vision Pro.app',
    icon='icon.icns' if Path('icon.icns').exists() else None,
    bundle_identifier='com.wifianalysis.wifivisionpro',
    info_plist={{
        'CFBundleShortVersionString': '{self.version}',
        'CFBundleVersion': '{self.version}',
        'CFBundleIdentifier': 'com.wifianalysis.wifivisionpro',
        'CFBundleName': 'WiFi Vision Pro',
        'CFBundleDisplayName': 'WiFi Vision Pro',
        'CFBundleInfoDictionaryVersion': '6.0',
        'CFBundlePackageType': 'APPL',
        'CFBundleSignature': 'WVPR',
        'NSRequiresAquaSystemAppearance': False,
        'NSHighResolutionCapable': True,
    }}
)
'''
    
    def create_nsis_script(self) -> str:
        """Create NSIS installer script"""
        return f'''# WiFi Vision Pro Installer Script
!define APP_NAME "WiFi Vision Pro"
!define APP_VERSION "{self.version}"
!define APP_PUBLISHER "WiFi Analysis Solutions"
!define APP_URL "https://github.com/wifianalysis/wifi-vision-pro"
!define APP_EXECUTABLE "WiFiVisionPro.exe"

# Installer settings
Name "${{APP_NAME}}"
OutFile "dist/WiFi_Vision_Pro_v{self.version}_Windows_Installer.exe"
InstallDir "$PROGRAMFILES64\\${{APP_NAME}}"
InstallDirRegKey HKLM "Software\\${{APP_PUBLISHER}}\\${{APP_NAME}}" ""
RequestExecutionLevel admin

# Modern UI
!include "MUI2.nsh"

# Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

# Languages
!insertmacro MUI_LANGUAGE "English"

# Installation section
Section "Install"
    SetOutPath "$INSTDIR"
    
    # Install files
    File /r "dist\\WiFiVisionPro\\*.*"
    
    # Create shortcuts
    CreateDirectory "$SMPROGRAMS\\${{APP_NAME}}"
    CreateShortcut "$SMPROGRAMS\\${{APP_NAME}}\\${{APP_NAME}}.lnk" "$INSTDIR\\${{APP_EXECUTABLE}}"
    CreateShortcut "$DESKTOP\\${{APP_NAME}}.lnk" "$INSTDIR\\${{APP_EXECUTABLE}}"
    
    # Registry entries
    WriteRegStr HKLM "Software\\${{APP_PUBLISHER}}\\${{APP_NAME}}" "" "$INSTDIR"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}" "DisplayName" "${{APP_NAME}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}" "UninstallString" "$INSTDIR\\Uninstall.exe"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}" "DisplayVersion" "${{APP_VERSION}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}" "Publisher" "${{APP_PUBLISHER}}"
    
    # Create uninstaller
    WriteUninstaller "$INSTDIR\\Uninstall.exe"
SectionEnd

# Uninstallation section
Section "Uninstall"
    # Remove files
    RMDir /r "$INSTDIR"
    
    # Remove shortcuts
    Delete "$DESKTOP\\${{APP_NAME}}.lnk"
    RMDir /r "$SMPROGRAMS\\${{APP_NAME}}"
    
    # Remove registry entries
    DeleteRegKey HKLM "Software\\${{APP_PUBLISHER}}\\${{APP_NAME}}"
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}"
SectionEnd
'''
    
    def create_windows_portable(self) -> bool:
        """Create Windows portable package"""
        print("Creating Windows portable package...")
        
        try:
            # Source directory
            source_dir = self.dist_dir / "WiFiVisionPro"
            if not source_dir.exists():
                print("PyInstaller output not found")
                return False
            
            # Create portable package
            portable_dir = self.dist_dir / f"WiFi_Vision_Pro_v{self.version}_Windows_Portable"
            if portable_dir.exists():
                shutil.rmtree(portable_dir)
            
            # Copy files
            shutil.copytree(source_dir, portable_dir)
            
            # Create run script
            run_script = portable_dir / "Run_WiFi_Vision_Pro.bat"
            with open(run_script, 'w') as f:
                f.write(f'''@echo off
echo Starting WiFi Vision Pro v{self.version}...
echo.
cd /d "%~dp0"
WiFiVisionPro.exe
pause
''')
            
            # Create README
            readme_file = portable_dir / "README_PORTABLE.txt"
            with open(readme_file, 'w') as f:
                f.write(f'''WiFi Vision Pro v{self.version} - Portable Edition

INSTALLATION:
This is a portable version - no installation required!

RUNNING THE APPLICATION:
1. Double-click "Run_WiFi_Vision_Pro.bat" to start the application
2. Or run "WiFiVisionPro.exe" directly

REQUIREMENTS:
- Windows 10/11 (64-bit)
- Administrator privileges for WiFi signal capture
- At least 2GB RAM
- WiFi adapter

FEATURES:
- Real-time WiFi signal visualization
- AI-powered signal analysis
- Cross-platform compatibility
- No installation required

For support, visit: https://github.com/wifianalysis/wifi-vision-pro
''')
            
            # Create ZIP archive
            zip_file = self.dist_dir / f"WiFi_Vision_Pro_v{self.version}_Windows_Portable.zip"
            with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(portable_dir):
                    for file in files:
                        file_path = Path(root) / file
                        arc_path = file_path.relative_to(portable_dir)
                        zf.write(file_path, arc_path)
            
            print(f"Windows portable package created: {zip_file}")
            return True
            
        except Exception as e:
            print(f"Windows portable package creation failed: {e}")
            return False
    
    def create_linux_appimage(self):
        """Create Linux AppImage"""
        print("Creating Linux AppImage...")
        
        try:
            # Create AppDir structure
            app_dir = self.build_dir / "WiFiVisionPro.AppDir"
            if app_dir.exists():
                shutil.rmtree(app_dir)
            
            app_dir.mkdir()
            (app_dir / "usr" / "bin").mkdir(parents=True)
            (app_dir / "usr" / "share" / "applications").mkdir(parents=True)
            (app_dir / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps").mkdir(parents=True)
            
            # Copy executable
            source_exe = self.dist_dir / "WiFiVisionPro" / "WiFiVisionPro"
            if source_exe.exists():
                shutil.copy2(source_exe, app_dir / "usr" / "bin" / "WiFiVisionPro")
                os.chmod(app_dir / "usr" / "bin" / "WiFiVisionPro", 0o755)
            
            # Create .desktop file
            desktop_content = f'''[Desktop Entry]
Type=Application
Name=WiFi Vision Pro
Exec=WiFiVisionPro
Icon=wifi-vision-pro
Comment=Advanced WiFi Signal Visualization with AI
Categories=Network;Utility;
Terminal=false
Version={self.version}
'''
            
            with open(app_dir / "usr" / "share" / "applications" / "wifi-vision-pro.desktop", 'w') as f:
                f.write(desktop_content)
            
            # Create AppRun script
            apprun_content = '''#!/bin/bash
HERE="$(dirname "$(readlink -f "${{0}}")")"
export APPDIR="$HERE"
export PATH="${HERE}/usr/bin:$PATH"
export LD_LIBRARY_PATH="${HERE}/usr/lib:$LD_LIBRARY_PATH"
cd "$HERE"
exec "$HERE/usr/bin/WiFiVisionPro" "$@"
'''
            
            with open(app_dir / "AppRun", 'w') as f:
                f.write(apprun_content)
            os.chmod(app_dir / "AppRun", 0o755)
            
            print("AppImage structure created")
            
        except Exception as e:
            print(f"AppImage creation warning: {e}")
    
    def create_linux_deb(self):
        """Create Debian package"""
        print("Creating Debian package...")
        
        try:
            # Create package structure
            package_dir = self.build_dir / f"wifi-vision-pro_{self.version}_amd64"
            if package_dir.exists():
                shutil.rmtree(package_dir)
            
            (package_dir / "DEBIAN").mkdir(parents=True)
            (package_dir / "usr" / "bin").mkdir(parents=True)
            (package_dir / "usr" / "share" / "applications").mkdir(parents=True)
            (package_dir / "usr" / "share" / "doc" / "wifi-vision-pro").mkdir(parents=True)
            
            # Control file
            control_content = f'''Package: wifi-vision-pro
Version: {self.version}
Section: net
Priority: optional
Architecture: amd64
Depends: python3, python3-pip, libqt5widgets5
Maintainer: WiFi Analysis Solutions <support@wifianalysis.com>
Description: Advanced WiFi Signal Visualization with AI
 WiFi Vision Pro provides real-time WiFi signal capture, analysis,
 and visualization using advanced AI models. Features include
 signal-to-image conversion, interference detection, and
 comprehensive network analysis.
'''
            
            with open(package_dir / "DEBIAN" / "control", 'w') as f:
                f.write(control_content)
            
            # Copy executable if exists
            source_exe = self.dist_dir / "WiFiVisionPro" / "WiFiVisionPro"
            if source_exe.exists():
                shutil.copy2(source_exe, package_dir / "usr" / "bin" / "wifi-vision-pro")
                os.chmod(package_dir / "usr" / "bin" / "wifi-vision-pro", 0o755)
            
            print("Debian package structure created")
            
        except Exception as e:
            print(f"Debian package creation warning: {e}")
    
    def create_macos_dmg(self):
        """Create macOS DMG installer"""
        print("Creating macOS DMG installer...")
        
        try:
            app_path = self.dist_dir / "WiFi Vision Pro.app"
            dmg_path = self.dist_dir / f"WiFi_Vision_Pro_v{self.version}_macOS.dmg"
            
            if not app_path.exists():
                print("macOS app bundle not found")
                return
            
            # Create temporary DMG directory
            dmg_dir = self.build_dir / "dmg_contents"
            if dmg_dir.exists():
                shutil.rmtree(dmg_dir)
            dmg_dir.mkdir()
            
            # Copy app to DMG directory
            shutil.copytree(app_path, dmg_dir / "WiFi Vision Pro.app")
            
            # Create Applications symlink
            os.symlink("/Applications", dmg_dir / "Applications")
            
            # Try to create DMG using hdiutil (macOS only)
            if platform.system() == "Darwin":
                cmd = [
                    "hdiutil", "create", "-volname", "WiFi Vision Pro",
                    "-srcfolder", str(dmg_dir),
                    "-ov", "-format", "UDZO",
                    str(dmg_path)
                ]
                subprocess.run(cmd, check=True)
                print(f"macOS DMG created: {dmg_path}")
            else:
                print("DMG creation requires macOS")
            
        except Exception as e:
            print(f"DMG creation warning: {e}")
    
    def create_version_info(self):
        """Create version info for Windows"""
        version_info = f'''# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({self.version.replace('.', ',')},0),
    prodvers=({self.version.replace('.', ',')},0),
    mask=0x3f,
    flags=0x0,
    OS=0x4,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          u'040904B0',
          [StringStruct(u'CompanyName', u'WiFi Analysis Solutions'),
           StringStruct(u'FileDescription', u'WiFi Vision Pro - Advanced Signal Visualization'),
           StringStruct(u'FileVersion', u'{self.version}'),
           StringStruct(u'InternalName', u'WiFiVisionPro'),
           StringStruct(u'LegalCopyright', u'© 2025 WiFi Analysis Solutions'),
           StringStruct(u'OriginalFilename', u'WiFiVisionPro.exe'),
           StringStruct(u'ProductName', u'WiFi Vision Pro'),
           StringStruct(u'ProductVersion', u'{self.version}')])
      ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
'''
        
        with open(self.project_dir / "version_info.txt", 'w') as f:
            f.write(version_info)
    
    def install_dependencies(self):
        """Install build dependencies"""
        print("Installing build dependencies...")
        
        try:
            # Install PyInstaller if not present
            subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller>=5.13.0"], check=True)
            
            # Install platform-specific dependencies
            if self.current_platform == "windows":
                print("Installing Windows dependencies...")
                # Windows-specific packages are in requirements.txt
                
            elif self.current_platform == "linux":
                print("Installing Linux dependencies...")
                subprocess.run([sys.executable, "-m", "pip", "install", "python-dbus", "pygobject"], check=True)
                
            elif self.current_platform == "darwin":
                print("Installing macOS dependencies...")
                subprocess.run([sys.executable, "-m", "pip", "install", "pyobjc-framework-Cocoa"], check=True)
            
            print("Dependencies installed")
            
        except Exception as e:
            print(f"Dependency installation warning: {e}")
    
    def create_build_info(self):
        """Create build information file"""
        build_info = {
            "app_name": self.app_name,
            "version": self.version,
            "build_date": datetime.now().isoformat(),
            "platform": self.current_platform,
            "python_version": sys.version,
            "build_system": "PyInstaller + Custom Scripts"
        }
        
        with open(self.dist_dir / f"build_info_{self.current_platform}.json", 'w') as f:
            json.dump(build_info, f, indent=2)

def main():
    """Main build script"""
    print("WiFi Vision Pro Cross-Platform Builder")
    print("=" * 50)
    
    builder = CrossPlatformBuilder()
    
    # Install dependencies
    builder.install_dependencies()
    
    # Create version info (Windows)
    if builder.current_platform == "windows":
        builder.create_version_info()
    
    # Build for current platform
    success = builder.build_for_current_platform()
    
    # Create build info
    builder.create_build_info()
    
    if success:
        print("\nBUILD COMPLETED SUCCESSFULLY!")
        print(f"Output directory: {builder.dist_dir}")
        print("\nGenerated files:")
        for file in builder.dist_dir.iterdir():
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.1f} MB)")
    else:
        print("\nBUILD FAILED!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())