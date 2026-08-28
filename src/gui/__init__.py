"""GeoTIFF orthomosaic inspector GUI (Streamlit) and its UI-agnostic core.

``raster_inspect`` and ``tiling_preview`` hold pure, headless logic (no Streamlit
imports) so they can be unit-tested and reused from any front end. ``app`` is the
thin Streamlit presentation layer.
"""
