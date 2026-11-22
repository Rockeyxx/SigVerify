; Windows installer for the packaged app (issue #42). Built by CI after PyInstaller:
;     ISCC.exe /DAppVersion=0.10.0 bundle\windows\sigverify.iss
; Per-user install, so no administrator password is needed. Uninstall removes the program
; folder only: user data in %USERPROFILE%\.sigverify is never touched.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
; Never change AppId: Windows uses it to find the installed version to update or remove.
AppId={{EDE42725-FF00-4178-B6CB-607F4F673152}
AppName=SigVerify
AppVersion={#AppVersion}
AppPublisher=EE 499 Team 06
PrivilegesRequired=lowest
DefaultDirName={autopf}\SigVerify
DisableProgramGroupPage=yes
OutputDir=..\..\dist
OutputBaseFilename=SigVerify-v{#AppVersion}-setup
UninstallDisplayIcon={app}\SigVerify.exe
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[InstallDelete]
; An update starts from a clean program folder, so libraries of the old version do not stay.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "..\..\dist\SigVerify\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\SigVerify"; Filename: "{app}\SigVerify.exe"
; Opens a console with every error and writes debug.log (user guide, troubleshooting).
Name: "{autoprograms}\SigVerify (debug)"; Filename: "{app}\SigVerify.exe"; Parameters: "--debug"

[Run]
Filename: "{app}\SigVerify.exe"; Description: "Start SigVerify"; Flags: nowait postinstall skipifsilent
