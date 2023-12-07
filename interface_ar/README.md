# Augmented Reality Hololens 1

This document guides you through the process of implementing the augmented reality (AR) interface featured in our case study with HoloLens (1st gen). To accomplish this, you'll create a Unity package using the provided folders and files.

Contents:
- [Interface Overview](#interface-overview)
- [Equipment Requirements](#equipment-requirements)
- [Software Requirements](#software-requirements)
- [Instructions](#instructions)

## Interface Overview
- How does it work?
  - The Commander script is designed to control the locations of holograms within a Unity app developed for the HoloLens device. The script acts as a central controller for managing holographic spheres in Unity, fetching data from a server, and updating the holographic display based on received information.
  - In our study, the spheres represent the waypoints for pickup and drop-off locations of the legs.
  - The initial coordinates of the spheres is read from the robotIniti.txt file (pulled from the server).
  - These coordinates correspond to the waypoints based on the initial set of training data the robot has been provided with.
  - As the human teacher provides corrections, the robot's training dataset is updated with the corrections, and the new coordinates of the waypoints based on this retraining is updated in the robotUpdate.txt file.
  - The Unity app then pulls these new coordinates from the server, and updates the holographic spheres coordinates in the AR environment.

## Equipment Requirements
- A Windows PC configured with the tools installed (see [Software Requirements](#software-requirements))
- A HoloLens (1st gen) device configured for development (visit [this article](https://learn.microsoft.com/en-us/windows/mixed-reality/develop/advanced-concepts/using-visual-studio?tabs=hl2#enabling-developer-mode) for more information).

  **NOTE:** these intructions are based on Wi-Fi deployment to HoloLens. Therefore, your PC and HoloLens device must be connected to the same network. A nice tool to verify Wi-Fi connectivity is the **Windows Device Portal**. Visit the [*Using the Windows Devive Portal* article](https://learn.microsoft.com/en-us/windows/mixed-reality/develop/advanced-concepts/using-the-windows-device-portal#connecting-over-wi-fi) for more information.


## Software Requirements
The following software versions were used to implement the AR environment:
- **Windows**
  - This app package was developed using Windows 11 Pro.
  - Enable developer mode on your PC at Settings > Update & Security > For developers
- **Visual Studio 2022**
  - In the Visual Studio Installer, locate your VS 2022 and click *Modify*. Make sure to install the following workloads:
      - .NET desktop development
      - Desktop development with C++
      - Universal Windows Platform (UWP) development
          - When selecting UWP, make sure to install Windows 11 SDK and C++ (v142) Universal Windows Platform tools
      - Game development with Unity
- **Unity Hub 3.6.0**
- **Unity 2019 LTS (2019.4.13f1)**
    - Make sure to install Universal Windows Platform Build Support while installing the Unity version



## Instructions
These instructions are based on [Microsoft Mixed Reality documentation](https://learn.microsoft.com/en-us/windows/mixed-reality/). We highly recommend visiting the *HoloLens (1st gen) Basics* [articles](https://learn.microsoft.com/en-us/windows/mixed-reality/develop/unity/tutorials/holograms-100) for detailed, step-by-step tutorials to get familiar with Mixed Reality development in Unity and development some quick holographic applications deployed in HoloLens 1. Following these tutorials guarantee your setup and tools are configured properly and you are familiar with development in Unity. This Microsoft documetation may be helpful to solve issues/questions you may encounter that are not addressed in these instructions. 

1. Create a new project in Unity
   - Open Unity Hub.
   - Log-in and click on *New Project*
   - Select *3D* template
   - Choose the right Unity version (2019.4.13f1)
   - Name your project and select its location in your drive
   - Click *Create project*
2. Import the provided files
   - Close the project, and navigate to the selected directory for your project.
   - Download the provided folders:
       - *Assets*
       - *Packages*
       - *ProjectSettings*
   - First, navigate to the *Assets* folder in your **project** directory. Select and delete all the contents. Then, copy the contents of the **downloaded** *Assets* folder, and paste it in your **project** *Assets* folder.
   - Repeat the previous step for the *Packages* and *ProjectSettings* folders.
   - Navigate to the *Assets* folder. Locate the *WSATestCertificate* file and meta file, and delete them. 
3. Relaunch the project in Unity Hub.
   - Now you should be able to see all assets and packages imported into the project.
4. Edit project settings
   - Navigate to Edit > Project Settings > Player
     - Change the Company and Product name as you wish.
     - Under *Publishing Settings*, change the Package name  and Description.
     - Every time you want to build a new version of an app package to debug, but wnat to keep old iterations of the app in the Hololens, you will need to change these names to make sure apps are not overwritten when uploaded to Hololens.
5. Build the app package
   - Navigate to File > Build Settings
   - Under *Scenes in Build*, **select *Scenes/project* only**
   - Select *Universal Windows Platform* under *Platform*
   - For the options, select:
       - Target Device: HoloLens
       - Architecture: x64
       - Build Type: D3D Project
       - Target SDK Version: Latest installed
       - Minimum Platform Version: 10.0.10240.0
       - Visual Studio Version: Latest installed
       - Build and Run on: Local Machine
       - Build configuration: Release
       - Checkmark *Development Build*
       - Compression Method: Default
   - Press *Switch Platform*
   - Wait for the status bar to complete, then press *Build*
   - A fie explorer window will pop-up, showing your project directory. Create a new folder named *App*. Open it, and then click *Select Folder*. A new status bar will show up. Wait for it to complete building the package. 
6. Deploy to Hololens using VS 2022
   - Navigate to the App folder you just created in your project directory when building the package.
   - Locate the *.sln* file that should be named as you indicated in project settings, and open it with Visual Studio 2022.
   - At this point, it is a good idea to open the **Windows Device Portal** to verify your Wi-Fi connectivity before trying deployment.
   - On the top bar of Visual Studio, select *Release*, *x86*, and *Remote Machine* from the dropdown menus.
   - Navigate to the Solution Explorer (usually on the right side of the screen), and select the file that has *(Universal Windows)* at the end of it. Right-click on it, and select *Properties*.
   -  A window will open. Select *Release* for Configuration, and *All Platforms* for Platform.
   -  Under *Configuration Properties*, select *Debugging*. Select *Remote Machine* for *Debugger to launch*. Write the IP address of your HoloLens device in the *Machine Name* field, and select *Yes* under *Deploy Visual C++ Debug Runtime Libraries*.
   -  Click OK
   -  Make sure the HoloLensdevice is turned on. In Visual Studio, press the green triangle (*Start Without Debugging*) to start deployment
   -  Supervise the Output console for any errors. Once the app is deployed, it will automatically start in the HoloLens device. 



