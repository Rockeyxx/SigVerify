"""
Tablet Capture Abstraction — Signature Verification System

Device-agnostic tablet capture layer with concrete driver implementations.
The abstract base class defines the interface; platform-specific drivers
implement the actual hardware communication.

Drivers:
  - WintabCapture:  Windows via Wintab API
  - EvdevCapture:   Linux via evdev/OpenTabletDriver

The interface supports any digitizer tablet — the XPPen Deco 640 is the
reference hardware but the abstraction is not device-specific.
"""

# TODO: Implement TabletCapture (ABC) with connect(), capture_signature(), get_capabilities()
# TODO: Implement WintabCapture (Windows Wintab API path)
# TODO: Implement EvdevCapture (Linux evdev/OpenTabletDriver path)
# TODO: Implement platform auto-detection and driver factory
# TODO: Implement software signature-box constraint on active area
