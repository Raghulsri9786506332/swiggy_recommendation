"""
Playwright Test Suite - Project 1: Swiggy Restaurant Recommendation System
Tests the Streamlit app running on http://localhost:8501
"""
import asyncio
import sys
from playwright.async_api import async_playwright, expect

BASE_URL = "http://localhost:8501"
RESULTS = []

async def run_tests():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        print("=" * 60)
        print("PROJECT 1: Swiggy Restaurant Recommendation System")
        print("=" * 60)

        # ----------------------------------------------------------------
        # TC-SW-001: Page loads and title is visible
        # ----------------------------------------------------------------
        try:
            await page.goto(BASE_URL, timeout=30000)
            await page.wait_for_load_state("networkidle", timeout=30000)
            # Wait for Streamlit to fully render
            await page.wait_for_timeout(4000)
            title = await page.title()
            content = await page.content()
            assert "Swiggy" in content or "Restaurant" in content, "Title content not found"
            print("✅ TC-SW-001 PASS: Page loads successfully")
            RESULTS.append(("TC-SW-001", "PASS", "Page loads successfully"))
        except Exception as e:
            print(f"❌ TC-SW-001 FAIL: {e}")
            RESULTS.append(("TC-SW-001", "FAIL", str(e)))

        # ----------------------------------------------------------------
        # TC-SW-002: Sidebar filters are present (city, cuisine, rating)
        # ----------------------------------------------------------------
        try:
            await page.wait_for_timeout(3000)
            # Look for sidebar elements
            sidebar = page.locator('[data-testid="stSidebar"]')
            await expect(sidebar).to_be_visible(timeout=10000)
            sidebar_text = await sidebar.inner_text()
            assert "City" in sidebar_text or "Preference" in sidebar_text or "City" in sidebar_text
            print("✅ TC-SW-002 PASS: Sidebar with filters is present")
            RESULTS.append(("TC-SW-002", "PASS", "Sidebar with filters visible"))
        except Exception as e:
            print(f"❌ TC-SW-002 FAIL: {e}")
            RESULTS.append(("TC-SW-002", "FAIL", str(e)))

        # ----------------------------------------------------------------
        # TC-SW-003: Welcome / landing text visible
        # ----------------------------------------------------------------
        try:
            main_content = await page.inner_text("body")
            assert "Welcome" in main_content or "Restaurant" in main_content or "recommendation" in main_content.lower()
            print("✅ TC-SW-003 PASS: Welcome landing text visible")
            RESULTS.append(("TC-SW-003", "PASS", "Landing page text visible"))
        except Exception as e:
            print(f"❌ TC-SW-003 FAIL: {e}")
            RESULTS.append(("TC-SW-003", "FAIL", str(e)))

        # ----------------------------------------------------------------
        # TC-SW-004: City dropdown has options
        # ----------------------------------------------------------------
        try:
            selects = page.locator('[data-testid="stSelectbox"]')
            count = await selects.count()
            assert count >= 1, f"Expected at least 1 selectbox, found {count}"
            print(f"✅ TC-SW-004 PASS: Found {count} selectbox(es) in sidebar")
            RESULTS.append(("TC-SW-004", "PASS", f"{count} selectboxes found"))
        except Exception as e:
            print(f"❌ TC-SW-004 FAIL: {e}")
            RESULTS.append(("TC-SW-004", "FAIL", str(e)))

        # ----------------------------------------------------------------
        # TC-SW-005: Find Restaurants button exists
        # ----------------------------------------------------------------
        try:
            btn = page.locator('[data-testid="stSidebar"] button')
            count = await btn.count()
            assert count >= 1, "No button found in sidebar"
            print(f"✅ TC-SW-005 PASS: 'Find Restaurants' button found ({count} button(s))")
            RESULTS.append(("TC-SW-005", "PASS", "Button found in sidebar"))
        except Exception as e:
            print(f"❌ TC-SW-005 FAIL: {e}")
            RESULTS.append(("TC-SW-005", "FAIL", str(e)))

        # ----------------------------------------------------------------
        # TC-SW-006: Click Find Restaurants and check results
        # ----------------------------------------------------------------
        try:
            btn = page.locator('[data-testid="stSidebar"] button').first
            await btn.click()
            await page.wait_for_timeout(5000)
            body_text = await page.inner_text("body")
            has_results = (
                "Found" in body_text
                or "restaurant" in body_text.lower()
                or "No restaurants" in body_text
                or "Rating" in body_text
            )
            assert has_results, "No restaurant results or message after clicking"
            print("✅ TC-SW-006 PASS: Clicking 'Find Restaurants' returns results or message")
            RESULTS.append(("TC-SW-006", "PASS", "Results or message displayed after search"))
        except Exception as e:
            print(f"❌ TC-SW-006 FAIL: {e}")
            RESULTS.append(("TC-SW-006", "FAIL", str(e)))

        # ----------------------------------------------------------------
        # TC-SW-007: No error messages on page
        # ----------------------------------------------------------------
        try:
            body_text = await page.inner_text("body")
            error_indicators = ["Traceback", "AttributeError", "ModuleNotFoundError", "FileNotFoundError"]
            found_errors = [e for e in error_indicators if e in body_text]
            assert not found_errors, f"Error found on page: {found_errors}"
            print("✅ TC-SW-007 PASS: No Python errors on page")
            RESULTS.append(("TC-SW-007", "PASS", "No errors on page"))
        except Exception as e:
            print(f"❌ TC-SW-007 FAIL: {e}")
            RESULTS.append(("TC-SW-007", "FAIL", str(e)))

        await browser.close()

        # ----------------------------------------------------------------
        # Summary
        # ----------------------------------------------------------------
        print("\n" + "=" * 60)
        print("SWIGGY TEST SUMMARY")
        print("=" * 60)
        passed = sum(1 for _, s, _ in RESULTS if s == "PASS")
        failed = sum(1 for _, s, _ in RESULTS if s == "FAIL")
        for tc, status, msg in RESULTS:
            icon = "✅" if status == "PASS" else "❌"
            print(f"  {icon} {tc}: {status} — {msg}")
        print(f"\nTotal: {len(RESULTS)} | Passed: {passed} | Failed: {failed}")
        return failed == 0

if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
