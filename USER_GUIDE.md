# SigVerify user guide

SigVerify checks a person's signature on this computer, from a photo of a paper
signature, a signature on a pen tablet, or both. Nothing is sent over the internet.

## Install

### Windows 10 or 11 (64-bit)

1. Download `SigVerify-vX.Y.Z-setup.exe` from the
   [Releases page](https://github.com/Rockeyxx/Signature-Verification-ML-EE499/releases).
2. Run it. The installer is not code-signed yet, so Windows may show "Windows
   protected your PC". Click **More info**, then **Run anyway**.
3. No administrator password is needed. The app is installed for your user only.
4. Start **SigVerify** from the Start menu.

### Linux (64-bit, tested on Arch)

1. Download `SigVerify-vX.Y.Z-x86_64.AppImage` from the
   [Releases page](https://github.com/Rockeyxx/Signature-Verification-ML-EE499/releases).
2. Allow it to run: right-click, Properties, "Allow executing file as program",
   or in a terminal `chmod +x SigVerify-*.AppImage`.
3. Double-click it.

If it does not open, the system is missing FUSE. Install `fuse3` (Arch:
`sudo pacman -S fuse3`), or run it without FUSE:
`./SigVerify-*.AppImage --appimage-extract-and-run`.

## First start

The first screen is **Create a vault**. A vault is one encrypted database with its
own owner, members and signers. Enter the vault name (your company or branch),
then the owner's full name, a User ID and a password.
If you already have a vault from another computer, choose **I already have a vault:
restore it from a backup** instead (see Backup and restore below).

Passwords are not stored anywhere. If the owner and every member forget their
passwords, the enrolled signatures cannot be recovered and every signer must
enrol again. Make a backup (see below).

After that, log in with your User ID and password. Only the owner adds members.

## Passwords

A new password needs at least 15 characters. Capitals, digits and symbols are not
required: a long phrase of ordinary words works. Very common passwords are refused.
The line under the password field says what is missing while you type. Passwords set
in an older version of the app still log in; the rule applies when a password is set
or changed.

**Generate password** fills both password fields with 20 random letters and digits and
shows them once, with a Copy button. Write the password down or keep it in a password
manager before you close that window: nobody can recover it, not even the owner.

The eye button at the end of a password field shows or hides what you typed. The text
is hidden again when you leave the page.

A member who forgets their password asks the owner to set a new one (Members page,
Reset Password). Anyone changes their own password on the Profile page, with the current one.

## Tablet setup

The app was tested with the XPPen Deco 640.

- **Windows:** install the XPPen driver before you enrol anyone. With the driver
  the app reads the pen through Wintab, with the full pressure range; without it,
  through Windows Ink. Keep the same setup afterwards: pen signatures enrolled
  with the driver do not match signatures made without it.
- **Linux:** no driver is needed.

Enrol and verify on the same tablet model and the same operating system. The app
refuses a pen signature from a different tablet or system and says so.

## Enrol a signer

1. Open **Enrollment** and enter a signer ID.
2. Add 5 photos of the person's signature (blue pen on white paper, phone held
   steady and close), 5 signatures on the tablet, or both.
3. Only the owner can replace a set the signer already has.

## Verify a signature

1. Open **Verification** and enter the signer ID.
2. Choose paper and pen, paper only, or pen only.
3. Give the photo and/or sign once on the tablet.
4. The app shows **Genuine** or **Forged** with the match score against the
   decision threshold. The score is a similarity, not a probability.

## Backup and restore

The owner opens **Members**, clicks **Back Up to a Folder** and picks a folder.
The backup file is encrypted like the vault.

To restore, the owner clicks **Restore from Backup** on the Members page. On a
new computer, use **I already have a vault: restore it from a backup** on the first screen instead,
then give the vault name and an owner's User ID and password from the backup.
The app keeps a copy of the current database before it restores, and you log in
again afterwards.

## Where your data is

The vaults are in the `.sigverify` folder in your home folder
(`%USERPROFILE%\.sigverify` on Windows, `~/.sigverify` on Linux). Updating or
uninstalling the app does not touch this folder.

## Messages and what to do

| Message says | What to do |
| --- | --- |
| The photo is not good enough to check (too small, too faint or too blurry) | Retake the photo on white paper with a blue pen, phone steady and close |
| The signature is too small in this photo | Take the photo closer, so the signature fills more of it |
| The pen signature could not be used (too short or not recorded properly) | Sign again on the tablet, inside the box, in one go |
| The pen references were signed on another tablet or system | Sign on the same tablet model and system as at enrolment, or ask the owner to re-enrol the pen signatures |
| The references were made by another model version | Ask the owner to re-enrol that signer (a new app version changed the model) |
| Refused: this signing was already used (replay) | The same signing was submitted twice. Sign again |
| Already has references. Only the owner can replace them | Ask the owner to re-enrol the signer |
| This database was made by an older version of the app | Move the folder named in the message somewhere else and start again; signers must enrol again |
| Log in first: the signature data is locked | Log in again |
| The password must have at least 15 characters | Use a longer password, or press Generate password |
| This password is too common | Choose another one, or press Generate password |
| Windows: `MSVCP140.dll` or `VCRUNTIME140.dll` was not found | Install the Microsoft Visual C++ Redistributable (x64) from Microsoft's website, then start the app again |

## If something goes wrong

Start the app in debug mode:

- **Windows:** Start menu, **SigVerify (debug)**. A console window shows every error.
- **Linux:** in a terminal, `./SigVerify-*.AppImage --debug`.

Errors are also saved to `debug.log` in the `.sigverify` folder. Send that file
with a short description of what you did. It holds error messages only, never
signatures or passwords.

## Uninstall

- **Windows:** Settings, Apps, Installed apps, **SigVerify**, Uninstall.
- **Linux:** delete the AppImage file (with AppImageLauncher: right-click SigVerify
  in the app menu, "Delete this AppImage").

Your vaults stay in the `.sigverify` folder. Delete that folder only if you want
to remove every enrolled signature for good.
