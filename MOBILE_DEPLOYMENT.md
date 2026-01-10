# Mobile Deployment Guide for SongBloom
# iOS and Android deployment instructions

## Overview

This guide provides comprehensive instructions for deploying SongBloom to iOS and Android platforms. The approach uses hybrid frameworks to maximize code reuse while maintaining native performance.

## Architecture

### Hybrid Framework Options

We recommend using **React Native** or **Flutter** for mobile deployment:

1. **React Native** (Recommended for JavaScript/TypeScript teams)
   - JavaScript/TypeScript based
   - Large ecosystem
   - Good native module support
   - Better for teams familiar with web development

2. **Flutter** (Recommended for optimal performance)
   - Dart-based
   - Excellent performance
   - Beautiful UI out of the box
   - Single codebase for iOS and Android

## iOS Deployment

### Prerequisites

- macOS with Xcode 14+
- Apple Developer Account ($99/year)
- CocoaPods installed
- iOS Deployment Target: 14.0+

### Step 1: Project Setup

#### Using React Native

```bash
# Install React Native CLI
npm install -g react-native-cli

# Create new project
npx react-native init SongBloomMobile --template react-native-template-typescript

# Navigate to iOS directory
cd SongBloomMobile/ios

# Install dependencies
pod install
```

#### Using Flutter

```bash
# Install Flutter SDK
# Follow: https://flutter.dev/docs/get-started/install

# Create new project
flutter create songbloom_mobile
cd songbloom_mobile
```

### Step 2: Configure iOS Project

#### Info.plist Configuration

Add required permissions to `ios/SongBloomMobile/Info.plist`:

```xml
<key>NSMicrophoneUsageDescription</key>
<string>SongBloom needs microphone access for voice cloning</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>SongBloom needs access to save generated music</string>
<key>NSAppleMusicUsageDescription</key>
<string>SongBloom can integrate with your music library</string>
```

#### Entitlements

Create `ios/SongBloomMobile.entitlements`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.app-sandbox</key>
    <false/>
    <key>com.apple.developer.associated-domains</key>
    <array>
        <string>applinks:songbloom.ai</string>
    </array>
</dict>
</plist>
```

### Step 3: Integrate Backend

Create API client for SongBloom backend:

```typescript
// services/api.ts
import axios from 'axios';

const API_BASE_URL = 'https://api.songbloom.ai/v1';

export class SongBloomAPI {
  private apiKey: string;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  async generateMusic(params: {
    lyrics: string;
    personaId?: string;
    stylePrompt?: File;
  }): Promise<{ jobId: string }> {
    const formData = new FormData();
    formData.append('lyrics', params.lyrics);
    if (params.personaId) {
      formData.append('persona_id', params.personaId);
    }
    if (params.stylePrompt) {
      formData.append('prompt_audio', params.stylePrompt);
    }

    const response = await axios.post(
      `${API_BASE_URL}/generate`,
      formData,
      {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  }

  async checkJobStatus(jobId: string): Promise<{
    status: string;
    result?: string;
  }> {
    const response = await axios.get(
      `${API_BASE_URL}/jobs/${jobId}`,
      {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
        },
      }
    );

    return response.data;
  }
}
```

### Step 4: Build and Sign

```bash
# Open Xcode
open ios/SongBloomMobile.xcworkspace

# In Xcode:
# 1. Select your team in Signing & Capabilities
# 2. Configure Bundle Identifier (com.yourcompany.songbloom)
# 3. Select Generic iOS Device as target
# 4. Product > Archive
# 5. Distribute App > App Store Connect
```

### Step 5: TestFlight Distribution

```bash
# Upload to App Store Connect
# Xcode will handle this in the Archive Organizer

# Or use command line with fastlane:
gem install fastlane

# Create Fastfile
fastlane init

# Add to Fastfile:
lane :beta do
  build_app(scheme: "SongBloomMobile")
  upload_to_testflight
end

# Deploy
fastlane beta
```

## Android Deployment

### Prerequisites

- Android Studio Arctic Fox or later
- Android SDK 24+ (Android 7.0)
- Java Development Kit (JDK) 11+
- Android device or emulator

### Step 1: Configure Android Project

#### build.gradle (app level)

```gradle
android {
    namespace 'com.songbloom.mobile'
    compileSdk 34

    defaultConfig {
        applicationId "com.songbloom.mobile"
        minSdk 24
        targetSdk 34
        versionCode 1
        versionName "3.0.0"
        
        // Multidex support
        multiDexEnabled true
    }

    buildTypes {
        release {
            minifyEnabled true
            shrinkResources true
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
            
            // Signing config
            signingConfig signingConfigs.release
        }
    }

    signingConfigs {
        release {
            storeFile file('keystore/songbloom-release.keystore')
            storePassword System.getenv("KEYSTORE_PASSWORD")
            keyAlias System.getenv("KEY_ALIAS")
            keyPassword System.getenv("KEY_PASSWORD")
        }
    }

    // Split APKs by ABI
    splits {
        abi {
            enable true
            reset()
            include 'armeabi-v7a', 'arm64-v8a', 'x86', 'x86_64'
            universalApk true
        }
    }
}

dependencies {
    implementation 'androidx.core:core-ktx:1.12.0'
    implementation 'androidx.appcompat:appcompat:1.6.1'
    implementation 'com.google.android.material:material:1.10.0'
    
    // Networking
    implementation 'com.squareup.retrofit2:retrofit:2.9.0'
    implementation 'com.squareup.retrofit2:converter-gson:2.9.0'
    
    // Audio
    implementation 'androidx.media3:media3-exoplayer:1.2.0'
    implementation 'androidx.media3:media3-ui:1.2.0'
}
```

#### AndroidManifest.xml

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">

    <!-- Permissions -->
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
    <uses-permission android:name="android.permission.RECORD_AUDIO" />
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE" />

    <application
        android:name=".SongBloomApplication"
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:roundIcon="@mipmap/ic_launcher_round"
        android:theme="@style/Theme.SongBloom"
        android:usesCleartextTraffic="false">
        
        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:launchMode="singleTask"
            android:theme="@style/Theme.SongBloom.Splash">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>

        <!-- Deep linking -->
        <activity
            android:name=".DeepLinkActivity"
            android:exported="true">
            <intent-filter android:autoVerify="true">
                <action android:name="android.intent.action.VIEW" />
                <category android:name="android.intent.category.DEFAULT" />
                <category android:name="android.intent.category.BROWSABLE" />
                <data
                    android:scheme="https"
                    android:host="songbloom.ai"
                    android:pathPrefix="/app" />
            </intent-filter>
        </activity>

    </application>

</manifest>
```

### Step 2: Generate Signing Key

```bash
# Generate keystore
keytool -genkey -v -keystore songbloom-release.keystore \
  -alias songbloom-key \
  -keyalg RSA \
  -keysize 2048 \
  -validity 10000

# Move to project
mkdir -p android/app/keystore
mv songbloom-release.keystore android/app/keystore/

# Create key.properties
cat > android/key.properties << EOF
storePassword=YOUR_STORE_PASSWORD
keyPassword=YOUR_KEY_PASSWORD
keyAlias=songbloom-key
storeFile=keystore/songbloom-release.keystore
EOF
```

### Step 3: Build APK/AAB

```bash
# Build APK
cd android
./gradlew assembleRelease

# Output: android/app/build/outputs/apk/release/app-release.apk

# Build App Bundle (for Play Store)
./gradlew bundleRelease

# Output: android/app/build/outputs/bundle/release/app-release.aab
```

### Step 4: Deploy to Play Store

```bash
# Using Fastlane
gem install fastlane

# Setup
cd android
fastlane init

# Add to Fastfile:
lane :deploy do
  gradle(task: 'clean bundleRelease')
  upload_to_play_store(
    track: 'internal',  # or 'beta', 'production'
    aab: 'app/build/outputs/bundle/release/app-release.aab'
  )
end

# Deploy
fastlane deploy
```

## Model Integration

### On-Device Models

For lightweight voice cloning models:

#### iOS (Core ML)

```python
# Convert to Core ML
import coremltools as ct

# Assuming you have a PyTorch model
model = load_voice_model()
traced_model = torch.jit.trace(model, example_input)

# Convert
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=input_shape)]
)

# Save
coreml_model.save("VoiceCloning.mlmodel")
```

#### Android (TensorFlow Lite)

```python
# Convert to TensorFlow Lite
import tensorflow as tf

# Convert model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('voice_cloning.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Server-Based Models

For heavy models (music generation), use API approach shown above.

## Testing

### iOS Testing

```bash
# Unit tests
xcodebuild test -workspace ios/SongBloomMobile.xcworkspace \
  -scheme SongBloomMobile -destination 'platform=iOS Simulator,name=iPhone 14'

# UI tests
xcodebuild test -workspace ios/SongBloomMobile.xcworkspace \
  -scheme SongBloomMobileUITests -destination 'platform=iOS Simulator,name=iPhone 14'
```

### Android Testing

```bash
# Unit tests
./gradlew test

# Instrumented tests
./gradlew connectedAndroidTest
```

## CI/CD Integration

See `.github/workflows/` for GitHub Actions examples:
- `ios-build.yml` - iOS build and deployment
- `android-build.yml` - Android build and deployment

## Distribution

### Direct APK Distribution

1. Build release APK
2. Host on your website or CDN
3. Users enable "Install from Unknown Sources"
4. Download and install APK

### Enterprise Distribution

#### iOS

1. Enroll in Apple Developer Enterprise Program
2. Create Enterprise Distribution certificate
3. Build with Enterprise profile
4. Distribute via MDM or direct download

#### Android

1. Build signed APK/AAB
2. Distribute via:
   - Your own website
   - Enterprise app store
   - MDM solution
   - Direct installation

## Monitoring & Analytics

Integrate analytics and crash reporting:

```typescript
// Initialize Firebase
import analytics from '@react-native-firebase/analytics';
import crashlytics from '@react-native-firebase/crashlytics';

// Track events
analytics().logEvent('music_generated', {
  persona_id: personaId,
  quality_preset: 'high',
});

// Log errors
crashlytics().log('Music generation failed');
```

## Security Best Practices

1. **API Key Management**: Use secure storage (Keychain/Keystore)
2. **Certificate Pinning**: Implement for API calls
3. **Code Obfuscation**: Enable ProGuard (Android) / BitCode (iOS)
4. **Secure Audio Storage**: Encrypt cached audio files
5. **Input Validation**: Sanitize all user inputs

## Resources

- [React Native Documentation](https://reactnative.dev/)
- [Flutter Documentation](https://flutter.dev/)
- [iOS Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/)
- [Android Material Design](https://material.io/design)
- [App Store Review Guidelines](https://developer.apple.com/app-store/review/guidelines/)
- [Google Play Policy](https://play.google.com/about/developer-content-policy/)
