import pandas as pd
from bs4 import BeautifulSoup
import cloudscraper
import time
import random
from datetime import datetime
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import queue
from contextlib import contextmanager
import atexit
import signal
import sys
import os
import psutil
import threading
import traceback

# ======================== CLEANUP MANAGEMENT ========================
# Global registry of active drivers for emergency cleanup
active_drivers = []
driver_registry_lock = Lock()

# Thread-safe print lock
print_lock = Lock()

# Driver creation lock to prevent race conditions
driver_creation_lock = Lock()

def safe_print(message):
    """Thread-safe printing with timestamp"""
    with print_lock:
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {message}")

def class_contains(class_attr, substring):
    """Return True if a BeautifulSoup class attribute contains a substring.

    Handles both string and list types that BeautifulSoup may provide for class_.
    """
    if not class_attr:
        return False
    if isinstance(class_attr, list):
        # Convert to string for searching
        class_str = ' '.join(str(item) for item in class_attr)
        return substring in class_str
    return isinstance(class_attr, str) and substring in class_attr

def register_driver(driver):
    """Register a driver for cleanup tracking"""
    with driver_registry_lock:
        active_drivers.append(driver)
        safe_print(f"   🔒 Registered driver (Total active: {len(active_drivers)})")

def unregister_driver(driver):
    """Remove a driver from registry after cleanup"""
    with driver_registry_lock:
        if driver in active_drivers:
            active_drivers.remove(driver)
            safe_print(f"   🔓 Unregistered driver (Total active: {len(active_drivers)})")

def cleanup_all_drivers():
    """Emergency cleanup of all registered drivers"""
    with driver_registry_lock:
        if active_drivers:
            safe_print(f"\n⚠️  Emergency cleanup: Closing {len(active_drivers)} active driver(s)...")
            for driver in active_drivers:
                try:
                    driver.quit()
                    safe_print(f"   ✅ Cleaned up driver session: {driver.session_id[:8]}...")
                except Exception as e:
                    safe_print(f"   ❌ Failed to cleanup driver: {str(e)[:50]}")
            active_drivers.clear()
            safe_print("   🧹 All drivers cleaned up")

def kill_orphaned_chrome_processes():
    """Kill all orphaned headless Chrome processes"""
    try:
        import psutil
        killed = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'chrome' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info.get('cmdline', []))
                    if 'headless' in cmdline and '--enable-automation' not in cmdline:
                        proc.terminate()
                        killed += 1
                        safe_print(f"   🔫 Terminated orphaned Chrome process: PID {proc.info['pid']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        if killed > 0:
            safe_print(f"   🧹 Killed {killed} orphaned Chrome process(es)")
    except ImportError:
        pass  # psutil not available

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    safe_print(f"\n\n🛑 Interrupt signal received (Signal: {signum})")
    safe_print("   Performing graceful shutdown...")
    cleanup_all_drivers()
    kill_orphaned_chrome_processes()
    safe_print("   👋 Goodbye!")
    sys.exit(0)

# Register cleanup handlers
atexit.register(cleanup_all_drivers)
atexit.register(kill_orphaned_chrome_processes)
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

# ======================== DRIVER MANAGEMENT ========================

def create_driver():
    """Creates a Chrome driver with anti-detection measures and cleanup tracking"""
    options = Options()
    
    # Anti-detection settings
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    # Performance settings
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--start-maximized")
    options.add_argument("--log-level=3")
    options.add_argument("--disable-logging")
    options.add_argument("--silent")
    
    # Memory optimization
    options.add_argument("--memory-pressure-off")
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-renderer-backgrounding")
    
    # User agent
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # Add a small delay between driver creations to avoid overwhelming the system
    with driver_creation_lock:
        time.sleep(0.5)
        try:
            driver = webdriver.Chrome(options=options)
        except Exception as e:
            safe_print(f"   ❌ Failed to create driver: {str(e)[:100]}")
            raise
    
    # Execute CDP commands to hide automation
    try:
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    except Exception:
        pass  # CDP commands might fail in some environments
    
    driver.implicitly_wait(10)
    return driver

@contextmanager
def selenium_driver(worker_id=None):
    """Enhanced context manager for safe driver creation and cleanup"""
    driver = None
    worker_str = f"Worker {worker_id}" if worker_id else "Main"
    
    try:
        safe_print(f"   🚗 [{worker_str}] Creating driver...")
        driver = create_driver()
        register_driver(driver)
        safe_print(f"   ✅ [{worker_str}] Driver ready")
        yield driver
    except Exception as e:
        safe_print(f"   ❌ [{worker_str}] Driver error: {str(e)[:100]}")
        raise
    finally:
        if driver:
            try:
                driver.quit()
                safe_print(f"   🔒 [{worker_str}] Driver closed successfully")
            except Exception as e:
                safe_print(f"   ⚠️ [{worker_str}] Failed to quit driver normally: {str(e)[:50]}")
                try:
                    # Force kill if normal quit fails
                    driver.service.process.terminate()
                    safe_print(f"   🔫 [{worker_str}] Force terminated driver process")
                except:
                    pass
            finally:
                unregister_driver(driver)

# ======================== WEB SCRAPING FUNCTIONS ========================

def parse_listing_page(html_doc):
    """Parses the search results page to extract property URLs and MLS numbers."""
    soup = BeautifulSoup(html_doc, 'html.parser')
    properties = []
    list_container = soup.find('div', id='listRow')
    if not list_container:
        safe_print("   ⚠️ Warning: Main listing container with id='listRow' not found.")
        return []

    listing_cards = list_container.find_all('div', class_=lambda c: c and 'styles___ListingPreview' in c)
    for card in listing_cards:
        link_tag = card.find('a', href=True)
        if link_tag and link_tag['href']:
            href = link_tag['href']
            if not href.startswith('http'):
                href = f"https://condos.ca{href}"

            if '/unit-' in href:
                card_text = card.get_text()
                mls_match = re.search(r'\b([CWENS]\d{7,9})\b', card_text)
                if not mls_match:
                    mls_match = re.search(r'MLS[®#:\s]*([A-Z]?\d{7,9})', card_text)

                mls_number = mls_match.group(1) if mls_match else None
                price_div = card.find('div', class_=lambda c: c and 'AskingPrice' in c)
                asking_price = price_div.text.strip() if price_div else None

                properties.append({
                    'url': href,
                    'mls_number': mls_number,
                    'asking_price': asking_price
                })
    return properties

def parse_listing_page_enhanced(html_doc):
    """Enhanced parsing with better error handling and debugging"""
    soup = BeautifulSoup(html_doc, 'html.parser')
    properties = []

    # Debug: Check if main container exists
    list_container = soup.find('div', id='listRow')
    if not list_container:
        safe_print("   ⚠️ Warning: Main listing container with id='listRow' not found.")
        # Try alternative selectors
        list_container = soup.find('div', class_=lambda c: c and 'ListingRow' in str(c))
        if not list_container:
            # Search the entire page as fallback
            list_container = soup
            safe_print("   ℹ️ Using entire page as search context")

    # Try multiple selectors for listing cards
    selectors = [
        lambda c: c and 'styles___ListingPreview' in c,
        lambda c: c and 'ListingPreview' in str(c),
        lambda c: c and 'listing-preview' in str(c).lower(),
    ]

    listing_cards = []
    for selector in selectors:
        listing_cards = list_container.find_all('div', class_=selector)
        if listing_cards:
            safe_print(f"   ℹ️ Found {len(listing_cards)} listing cards with selector")
            break

    if not listing_cards:
        # Try finding any links to /unit- pages as a last resort
        unit_links = soup.find_all('a', href=lambda h: h and '/unit-' in h)
        safe_print(f"   ℹ️ Found {len(unit_links)} unit links as fallback")

        for link in unit_links:
            href = link['href']
            if not href.startswith('http'):
                href = f"https://condos.ca{href}"

            # Try to find MLS and price in parent elements
            parent = link.parent
            while parent and parent.name != 'body':
                parent_text = parent.get_text()
                mls_match = re.search(r'\b([CWENS]\d{7,9})\b', parent_text)
                price_match = re.search(r'\$[\d,]+', parent_text)

                if mls_match or price_match:
                    properties.append({
                        'url': href,
                        'mls_number': mls_match.group(1) if mls_match else None,
                        'asking_price': price_match.group(0) if price_match else None
                    })
                    break
                parent = parent.parent
    else:
        # Original parsing logic
        for card in listing_cards:
            link_tag = card.find('a', href=True)
            if link_tag and link_tag['href']:
                href = link_tag['href']
                if not href.startswith('http'):
                    href = f"https://condos.ca{href}"

                if '/unit-' in href:
                    card_text = card.get_text()
                    mls_match = re.search(r'\b([CWENS]\d{7,9})\b', card_text)
                    if not mls_match:
                        mls_match = re.search(r'MLS[®#:\s]*([A-Z]?\d{7,9})', card_text)

                    mls_number = mls_match.group(1) if mls_match else None
                    price_div = card.find('div', class_=lambda c: c and 'AskingPrice' in c)
                    asking_price = price_div.text.strip() if price_div else None

                    properties.append({
                        'url': href,
                        'mls_number': mls_number,
                        'asking_price': asking_price
                    })

    # Remove duplicates based on URL
    seen_urls = set()
    unique_properties = []
    for prop in properties:
        if prop['url'] not in seen_urls:
            seen_urls.add(prop['url'])
            unique_properties.append(prop)

    return unique_properties

def extract_price_history(soup):
    """Extracts price history for both sales and leases."""
    price_history = []
    
    # Find history section - look for either the collapsible container or the history card container directly
    history_section = None
    
    # First try to find the main price history container
    history_section = soup.find('div', class_=lambda c: c and 'ListingPriceHistoryContainer' in str(c))
    
    # If not found, try the history card container directly
    if not history_section:
        history_section = soup.find('div', class_=lambda c: c and 'HistoryCardContainer' in str(c))
    
    # If still not found, try the archive list container
    if not history_section:
        history_section = soup.find('div', class_=lambda c: c and 'ArchiveListContainer' in str(c))
    
    # As a last resort, search the entire document
    if not history_section:
        history_section = soup
    
    # Find all archive cards (history entries)
    history_cards = history_section.find_all('a', class_=lambda c: c and 'ArchiveCard' in str(c))
    
    if not history_cards:
        # Try finding them without the 'a' tag restriction
        history_cards = history_section.find_all('div', class_=lambda c: c and 'ArchiveCard' in str(c))
    
    for card in history_cards:
        event = {}
        
        # Extract date - look for Date-sc class
        date_div = card.find('div', class_=lambda c: c and 'Date-sc' in str(c))
        if date_div:
            event['date'] = date_div.text.strip()
        
        # Extract status from Status-sc div
        status_div = card.find('div', class_=lambda c: c and 'Status-sc' in str(c))
        if status_div:
            status_text = status_div.get_text(strip=True)

            # Determine event type from the status text
            if 'Sold' in status_text:
                event['type'] = 'Sold'
                event['listing_type'] = 'Sale'  # Explicitly mark as sale
                # Extract sold price - try multiple methods
                sold_price = None
                price_match = re.search(r'Sold\s*for\s*\$?([\d,]+)', status_text)
                if price_match:
                    sold_price = price_match.group(1).replace(',', '')
                else:
                    # Try to find price in BlurCont divs within status
                    blur_divs = status_div.find_all('div', class_=lambda c: c and 'BlurCont' in str(c))
                    for blur_div in blur_divs:
                        blur_text = blur_div.get_text(strip=True)
                        if blur_text and '$' in blur_text:
                            price_match = re.search(r'\$?([\d,]+)', blur_text)
                            if price_match:
                                sold_price = price_match.group(1).replace(',', '')
                                break
                if sold_price:
                    event['sold_price'] = sold_price

            elif 'Leased' in status_text:
                event['type'] = 'Leased'
                event['listing_type'] = 'Lease'  # Explicitly mark as lease
                # Extract lease price - try multiple methods
                leased_price = None
                price_match = re.search(r'Leased\s*for\s*\$?([\d,]+)', status_text)
                if price_match:
                    leased_price = price_match.group(1).replace(',', '')
                else:
                    # Try to find price in BlurCont divs within status
                    blur_divs = status_div.find_all('div', class_=lambda c: c and 'BlurCont' in str(c))
                    for blur_div in blur_divs:
                        blur_text = blur_div.get_text(strip=True)
                        if blur_text and '$' in blur_text:
                            price_match = re.search(r'\$?([\d,]+)', blur_text)
                            if price_match:
                                leased_price = price_match.group(1).replace(',', '')
                                break
                if leased_price:
                    event['leased_price'] = leased_price

            elif 'Terminated' in status_text:
                event['type'] = 'Terminated'
            elif 'Expired' in status_text:
                event['type'] = 'Expired'
            elif 'Listed' in status_text:
                event['type'] = 'Listed'
            else:
                event['type'] = status_text.strip()
        
        # Extract listing price from ListedInfo section
        listed_info = card.find('div', class_=lambda c: c and 'ListedInfo' in str(c))
        if listed_info:
            # Find all BlurCont divs which contain the actual data
            blur_divs = listed_info.find_all('div', class_=lambda c: c and 'BlurCont' in str(c))
            
            if blur_divs:
                # First blur div contains the price
                if len(blur_divs) > 0:
                    price_text = blur_divs[0].text.strip()
                    # Clean the price - remove $, commas, and any non-numeric characters except digits
                    clean_price = re.sub(r'[^\d]', '', price_text)
                    if clean_price:
                        event['listed_price'] = clean_price
                        
                        # Determine if this is a sale or lease based on price
                        # Generally, prices under 10,000 are monthly rentals
                        price_value = int(clean_price)
                        if price_value < 10000:
                            event['listing_type'] = event.get('listing_type', 'Lease')
                        else:
                            event['listing_type'] = event.get('listing_type', 'Sale')
                
                # Second blur div contains the listing date
                if len(blur_divs) > 1:
                    event['listing_date'] = blur_divs[1].text.strip()
            else:
                # Fallback: try to extract price from the full text
                listed_text = listed_info.get_text(strip=True)
                # Look for price pattern
                price_match = re.search(r'\$?([\d,]+)', listed_text)
                if price_match:
                    clean_price = price_match.group(1).replace(',', '')
                    event['listed_price'] = clean_price
                    
                    # Determine if sale or lease based on price
                    try:
                        price_value = int(clean_price)
                        if price_value < 10000:
                            event['listing_type'] = event.get('listing_type', 'Lease')
                        else:
                            event['listing_type'] = event.get('listing_type', 'Sale')
                    except:
                        pass
                        
                # Look for date pattern
                date_match = re.search(r'on\s*([A-Za-z]+\s+\d+,?\s+\d{4})', listed_text)
                if date_match:
                    event['listing_date'] = date_match.group(1)
        
        # Extract days on market
        dom_div = card.find('div', class_=lambda c: c and 'DaysOnMarket' in str(c))
        if dom_div:
            # Look for BlurCont div with the number
            blur_div = dom_div.find('div', class_=lambda c: c and 'BlurCont' in str(c))
            if blur_div:
                dom_text = blur_div.text.strip()
                if dom_text.isdigit():
                    event['days_on_market'] = dom_text
            else:
                # Fallback: extract number from full text
                dom_text = dom_div.get_text(strip=True)
                dom_match = re.search(r'(\d+)', dom_text)
                if dom_match:
                    event['days_on_market'] = dom_match.group(1)
        
        # Add event to history if it has meaningful data
        if event and (event.get('type') or event.get('listed_price')):
            price_history.append(event)
    
    # If no history found, return empty list
    return price_history

def extract_preview_listings(soup, section_id):
    """Extract data from listing preview cards."""
    listings = []
    section = soup.find('div', id=section_id)
    if not section:
        return listings

    search_context = section
    if section_id == 'NearbyListings':
        map_container = section.find('div', class_=lambda c: c and 'MapPreviewsContainer' in c)
        if map_container:
            search_context = map_container

    listing_cards = search_context.find_all('div', class_=lambda c: c and 'ListingPreview-sc' in c)
    
    for card in listing_cards:
        listing_data = {}
        link_tag = card.find('a', class_=lambda c: c and 'LinkWrapper-sc' in c)
        if link_tag and link_tag.has_attr('href'):
            listing_data['url'] = f"https://condos.ca{link_tag['href']}"
        
        address_tag = card.find('address', class_=lambda c: c and 'Address-sc' in c)
        if address_tag:
            listing_data['address'] = address_tag.text.strip()
            
        price_tag = card.find('div', class_=lambda c: c and 'AskingPrice-sc' in c)
        if price_tag:
            price_text = price_tag.text.strip()
            listing_data['price'] = price_text.split(' ')[0] if price_text else 'N/A'
            
        info_holder = card.find('div', class_=lambda c: c and 'InfoHolder-sc' in c)
        if info_holder:
            details = info_holder.find_all('div', class_=lambda c: c and 'InfoItem-sc' in c)
            listing_data['beds'] = details[0].text.strip() if len(details) > 0 else 'N/A'
            listing_data['baths'] = details[1].text.strip() if len(details) > 1 else 'N/A'
            listing_data['parking'] = details[2].text.strip() if len(details) > 2 else 'N/A'
            sqft_tag = info_holder.find('div', class_=lambda c: c and 'ExactSize-sc' in c)
            listing_data['sqft'] = sqft_tag.text.strip() if sqft_tag else 'N/A'
        
        mls_tag = card.find('div', class_=lambda c: c and 'Mls-sc' in c)
        if mls_tag:
            listing_data['mls_number'] = mls_tag.text.replace('MLS#:', '').strip()

        if listing_data and listing_data.get('address'):
            listings.append(listing_data)
            
    return listings

def wait_for_page_load(driver, timeout=20):
    """Wait for page to be fully loaded"""
    try:
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        time.sleep(2)
        
        # Scroll to trigger lazy loading
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/3);")
        time.sleep(0.5)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight*2/3);")
        time.sleep(0.5)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.5)
        
        return True
    except:
        return False

def extract_nearby_listings_with_tabs(driver):
    """Extract both For Sale and For Rent listings from the Nearby Listings section"""
    nearby_data = {
        'for_sale': [],
        'for_rent': []
    }
    
    try:
        # Find nearby listings section
        nearby_element = None
        try:
            nearby_element = driver.find_element(By.ID, "NearbyListings")
        except:
            try:
                nearby_element = driver.find_element(By.XPATH, "//h2[contains(text(), 'Nearby Listings')]")
            except:
                return nearby_data
        
        if nearby_element:
            driver.execute_script("arguments[0].scrollIntoView(true);", nearby_element)
            time.sleep(1)
        
        # Get sale listings (default tab)
        sale_soup = BeautifulSoup(driver.page_source, 'html.parser')
        nearby_data['for_sale'] = extract_preview_listings(sale_soup, 'NearbyListings')
        
        # Try to click on "For Rent" tab
        try:
            rent_button = None
            for selector in [
                "//div[@id='NearbyListings']//button[contains(@aria-label, 'For Rent')]",
                "//div[@id='NearbyListings']//button[contains(., 'For Rent')]",
                "//button[contains(text(), 'For Rent')]"
            ]:
                try:
                    rent_button = driver.find_element(By.XPATH, selector)
                    break
                except:
                    continue
            
            if rent_button:
                driver.execute_script("arguments[0].scrollIntoView(true);", rent_button)
                time.sleep(0.5)
                driver.execute_script("arguments[0].click();", rent_button)
                time.sleep(2)
                
                rent_soup = BeautifulSoup(driver.page_source, 'html.parser')
                nearby_data['for_rent'] = extract_preview_listings(rent_soup, 'NearbyListings')
        except:
            pass
            
    except:
        pass
    
    return nearby_data

def extract_condo_details_selenium(driver, url, mls_number=None):
    """Parse page using Selenium driver to extract detailed information"""
    try:
        driver.get(url)
        wait_for_page_load(driver)
        
        html_content = driver.page_source
        
        if "Access Denied" in html_content or "403 Forbidden" in html_content:
            return None
            
        soup = BeautifulSoup(html_content, 'html.parser')
        
        condo_data = {
            'url': url, 'mls_number': mls_number or 'N/A', 'address': 'N/A', 'unit_number': 'N/A',
            'building_name': 'N/A', 'neighbourhood': 'N/A', 'price': 'N/A', 'beds': 'N/A',
            'baths': 'N/A', 'parking': 'N/A', 'sqft': 'N/A', 'property_type': 'N/A',
            'maintenance_fees': 'N/A', 'taxes': 'N/A', 'age_of_building': 'N/A',
            'outdoor_space': 'N/A', 'days_on_market': 'N/A', 'listing_date': 'N/A',
            'brokerage': 'N/A', 'description': 'N/A', 'amenities': 'N/A',
            'price_history_for_sale': 'N/A', 'price_history_for_lease': 'N/A',
            'similar_listings': 'N/A',
            'nearby_listings_for_sale': 'N/A',
            'nearby_listings_for_rent': 'N/A',
            'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Extract all details
        title = soup.find('h1', class_=lambda c: c and '___Title-sc-' in c)
        if title:
            condo_data['address'] = title.text.strip()
            unit_match = re.match(r'([\w-]+)\s*-', condo_data['address'])
            if unit_match: 
                condo_data['unit_number'] = unit_match.group(1)

        location_div = soup.find('div', class_=lambda c: c and '___Address-sc-' in c)
        if location_div:
            parts = [a.text.strip() for a in location_div.find_all('a')]
            if len(parts) > 1:
                condo_data['building_name'] = parts[0]
                condo_data['neighbourhood'] = parts[1]
        
        price_div = soup.find('div', class_=lambda c: c and '___Price-sc-' in c)
        if price_div: 
            condo_data['price'] = price_div.text.strip()

        dom_div = soup.find('div', class_=lambda c: c and '___ListedAgo-sc' in c)
        if dom_div: 
            condo_data['days_on_market'] = dom_div.text.strip()

        details_div = soup.find('div', class_=lambda c: c and '___Details-sc-ka5njm-10' in c)
        if details_div:
            if beds_span := details_div.find('div', class_=lambda c: c and 'BedDetails' in c): 
                condo_data['beds'] = beds_span.text.strip()
            if baths_span := details_div.find('div', class_=lambda c: c and 'BathDetails' in c): 
                condo_data['baths'] = baths_span.text.strip()
            if parking_span := details_div.find('div', class_=lambda c: c and 'ParkingDetails' in c): 
                condo_data['parking'] = parking_span.text.strip()
            if sqft_span := details_div.find('div', class_=lambda c: c and 'SqftDetails' in c): 
                condo_data['sqft'] = sqft_span.text.strip()
            if type_span := details_div.find('div', class_=lambda c: c and 'PropertyTypeDetails' in c): 
                condo_data['property_type'] = type_span.text.strip()

        details_section = soup.find('div', id='KeyFactsAndListingDetails')
        if details_section:
            for pair in details_section.find_all('div', class_=lambda c: c and 'TitleValueBlockV2' in c):
                if (title_div := pair.find('div', class_=lambda c: c and 'InfoRowTitle' in c)) and \
                   (value_div := pair.find('div', class_=lambda c: c and 'InfoRowValue' in c)):
                    title_text = title_div.text.strip().lower()
                    value_text = value_div.text.strip()
                    if 'maintenance' in title_text: 
                        condo_data['maintenance_fees'] = value_text
                    elif 'taxes' in title_text: 
                        condo_data['taxes'] = value_text
                    elif 'age of building' in title_text: 
                        condo_data['age_of_building'] = value_text
                    elif 'outdoor space' in title_text: 
                        condo_data['outdoor_space'] = value_text
                    elif 'listed on' in title_text: 
                        condo_data['listing_date'] = value_text
            
            if desc_div := details_section.find('div', class_=lambda c: c and 'BodyHtml' in c):
                condo_data['description'] = " ".join(desc_div.text.strip().split())
            
            if brokerage_div := details_section.find('div', class_=lambda c: c and 'BrokrageAndMLSCont' in c):
                if brokerage_text := brokerage_div.find(string=re.compile("Broker:")): 
                    condo_data['brokerage'] = brokerage_text.replace('Broker:', '').strip()
                if not condo_data['mls_number'] or condo_data['mls_number'] == 'N/A':
                    if mls_text := brokerage_div.find(string=re.compile("MLS")): 
                        condo_data['mls_number'] = mls_text.split(':')[-1].strip()

        if amenities_section := soup.find('div', id='Amenities'):
            amenities = [div.text.strip() for div in amenities_section.find_all('div', class_=lambda c: c and 'Amenity-sc' in c)]
            condo_data['amenities'] = ', '.join(amenities) if amenities else 'N/A'
            
        # Ensure Price History section is expanded and loaded
        try:
            price_history_anchor = None
            for xpath in [
                "//h2[contains(., 'Price History')]",
                "//label[h2[contains(., 'Price History')]]",
                "//input[contains(@id,'PriceHistory')]"
            ]:
                try:
                    price_history_anchor = driver.find_element(By.XPATH, xpath)
                    if price_history_anchor:
                        break
                except Exception:
                    continue

            if price_history_anchor:
                driver.execute_script("arguments[0].scrollIntoView(true);", price_history_anchor)
                time.sleep(0.8)
                try:
                    # If it's a checkbox and not checked, toggle via its label
                    if price_history_anchor.tag_name.lower() == 'input':
                        is_checked = price_history_anchor.is_selected()
                        if not is_checked:
                            # Try clicking associated label
                            input_id = price_history_anchor.get_attribute('id')
                            if input_id:
                                try:
                                    label = driver.find_element(By.XPATH, f"//label[@for='{input_id}']")
                                    driver.execute_script("arguments[0].click();", label)
                                except Exception:
                                    driver.execute_script("arguments[0].click();", price_history_anchor)
                            else:
                                driver.execute_script("arguments[0].click();", price_history_anchor)
                    else:
                        # Try clicking the enclosing label if present
                        try:
                            label = price_history_anchor.find_element(By.XPATH, "ancestor::label[1]")
                            driver.execute_script("arguments[0].click();", label)
                        except Exception:
                            pass
                except Exception:
                    pass

                time.sleep(1.0)
                soup = BeautifulSoup(driver.page_source, 'html.parser')
        except Exception:
            pass

        full_history = extract_price_history(soup)
        # Separate sale and lease history based on listing_type
        sale_history = [h for h in full_history if h.get('listing_type') == 'Sale']
        lease_history = [h for h in full_history if h.get('listing_type') == 'Lease']
        
        condo_data['price_history_for_sale'] = json.dumps(sale_history)
        condo_data['price_history_for_lease'] = json.dumps(lease_history)
        condo_data['similar_listings'] = json.dumps(extract_preview_listings(soup, 'SimilarListings'))
        
        # Get both sale and rental nearby listings
        nearby_data = extract_nearby_listings_with_tabs(driver)
        condo_data['nearby_listings_for_sale'] = json.dumps(nearby_data['for_sale'])
        condo_data['nearby_listings_for_rent'] = json.dumps(nearby_data['for_rent'])

        return condo_data

    except Exception as e:
        safe_print(f"      ⚠️ Extract error: {str(e)[:100]}")
        return None

def fetch_page(url, delay_range=(0.5, 1.5)):
    time.sleep(random.uniform(*delay_range))
    scraper = cloudscraper.create_scraper()
    try:
        response = scraper.get(url, timeout=60)
        if response.status_code == 200:
            return response.text
    except:
        pass
    return None

def fetch_search_page(args):
    page_num, url = args
    safe_print(f"📄 Fetching search page {page_num}...")
    html_content = fetch_page(url, delay_range=(0.2, 0.5))
    if html_content:
        properties = parse_listing_page(html_content)
        if properties:
            safe_print(f"   ✅ Page {page_num}: Found {len(properties)} properties.")
            return page_num, properties
        else:
            safe_print(f"   ⚠️ Page {page_num}: No properties found.")
            return page_num, []
    else:
        safe_print(f"   ❌ Page {page_num}: Failed to fetch.")
        return page_num, []

def fetch_search_page_selenium(args):
    """Fetch search page using Selenium for better JavaScript handling"""
    page_num, url = args
    worker_id = threading.get_ident() % 1000
    safe_print(f"🔄 [W{worker_id:03d}] Fetching search page {page_num} with Selenium...")

    max_retries = 3
    expected_min_properties = 40  # Set a minimum threshold (e.g., 40 out of 52)

    for retry in range(max_retries):
        try:
            with selenium_driver(worker_id) as driver:
                driver.get(url)

                # Wait for the listing container to be present
                try:
                    WebDriverWait(driver, 20).until(
                        EC.presence_of_element_located((By.ID, "listRow"))
                    )
                except TimeoutException:
                    safe_print(f"   ⚠️ Page {page_num}: Listing container not found, retrying...")
                    continue

                # Wait for property cards to load
                try:
                    # Wait until we have a reasonable number of property cards
                    WebDriverWait(driver, 15).until(
                        lambda d: len(d.find_elements(By.CSS_SELECTOR,
                            "div[class*='styles___ListingPreview']")) >= expected_min_properties
                    )
                except TimeoutException:
                    # Even if timeout, continue to see what we got
                    pass

                # Additional wait for any lazy-loaded content
                time.sleep(2)

                # Scroll to trigger any lazy loading
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
                time.sleep(1)
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)

                # Get the page source after everything is loaded
                html_content = driver.page_source
                properties = parse_listing_page_enhanced(html_content)

                if len(properties) >= expected_min_properties:
                    safe_print(f"   ✅ Page {page_num}: Found {len(properties)} properties (expected ~52)")
                    return page_num, properties
                elif properties:
                    safe_print(f"   ⚠️ Page {page_num}: Only found {len(properties)} properties, retrying... (attempt {retry + 1}/{max_retries})")
                    if retry == max_retries - 1:
                        # On last retry, return what we have
                        safe_print(f"   ⚠️ Page {page_num}: Returning {len(properties)} properties after {max_retries} attempts")
                        return page_num, properties
                else:
                    safe_print(f"   ⚠️ Page {page_num}: No properties found, retrying... (attempt {retry + 1}/{max_retries})")

        except Exception as e:
            safe_print(f"   ❌ Page {page_num}: Error on attempt {retry + 1}: {str(e)[:100]}")
            if retry == max_retries - 1:
                # Fall back to CloudScraper on final failure
                safe_print(f"   🔄 Page {page_num}: Falling back to CloudScraper...")
                return fetch_search_page(args)

    return page_num, []

def process_property_with_driver(args):
    """Process a single property with its own driver instance"""
    idx, total, property_dict = args
    property_url = property_dict['url']
    mls_number = property_dict.get('mls_number')
    worker_id = threading.get_ident() % 1000
    
    safe_print(f"  [W{worker_id:03d}] [{idx:>3}/{total}] Starting: {property_url.split('/')[-1][:40]}...")
    
    try:
        with selenium_driver(worker_id) as driver:
            details = extract_condo_details_selenium(driver, property_url, mls_number)
            if details and details.get('address') != 'N/A':
                safe_print(f"  [W{worker_id:03d}] [{idx:>3}/{total}] ✅ Success | {details.get('address', 'Unknown')[:40]} - {details.get('price', 'N/A')}")
                return details
            else:
                safe_print(f"  [W{worker_id:03d}] [{idx:>3}/{total}] ⚠️ Failed to parse details")
                return None
    except Exception as e:
        safe_print(f"  [W{worker_id:03d}] [{idx:>3}/{total}] ❌ Error: {str(e)[:100]}")
        if "--debug" in sys.argv:
            safe_print(f"     Stack trace:\n{traceback.format_exc()}")
        return None

def process_properties_parallel(properties, max_workers=3):
    """Process properties in parallel with multiple driver instances"""
    all_condos = []
    total = len(properties)
    
    # Create tasks with index information
    tasks = [(i, total, prop) for i, prop in enumerate(properties, 1)]
    
    safe_print(f"\n   🚀 Starting parallel processing with {max_workers} workers...")
    safe_print(f"   📊 Total properties to process: {total}")
    safe_print(f"   🔒 Active drivers will be tracked for cleanup\n")
    
    completed = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_property_with_driver, task): task for task in tasks}
        
        # Process completed futures
        for future in as_completed(futures):
            try:
                result = future.result(timeout=60)  # 60 second timeout per property
                if result:
                    all_condos.append(result)
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                task = futures[future]
                safe_print(f"   ❌ Failed to process property {task[0]}: {str(e)[:100]}")
                failed += 1
            
            # Progress update
            if (completed + failed) % 10 == 0:
                safe_print(f"   📊 Progress: {completed + failed}/{total} processed ({completed} success, {failed} failed)")
            
            # Add a small random delay between completions to avoid overwhelming the site
            time.sleep(random.uniform(0.5, 1.5))
    
    safe_print(f"\n   ✅ Parallel processing complete")
    safe_print(f"   📊 Final: {completed}/{total} successful, {failed}/{total} failed")
    return all_condos

def scrape_condos_fully_parallel(base_url, start_page=1, max_pages=5, max_search_workers=5, max_detail_workers=3):
    """Use parallel processing for both search pages and property details with robust cleanup"""
    safe_print(f"\n🚀 Starting FULLY PARALLEL scraping at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print(f"   Search: {max_search_workers} parallel workers")
    safe_print(f"   Details: {max_detail_workers} parallel workers with individual Selenium instances")
    safe_print(f"   🛡️ Robust cleanup enabled\n" + "=" * 60)

    search_tasks = [(page, base_url.replace('page=2', f'page={page}')) for page in range(start_page, start_page + max_pages)]

    safe_print(f"\n📚 Phase 1: Fetching {len(search_tasks)} search result pages...\n" + "-" * 60)
    all_properties = []

    start_time_search = time.time()
    with ThreadPoolExecutor(max_workers=max_search_workers) as executor:
        futures = [executor.submit(fetch_search_page, task) for task in search_tasks]
        for future in as_completed(futures):
            try:
                _, properties = future.result()
                if properties:
                    all_properties.extend(properties)
            except Exception as e:
                safe_print(f"   ❌ Exception: {e}")

    search_elapsed = time.time() - start_time_search

    unique_urls = {prop['url']: prop for prop in all_properties}.values()
    unique_urls = list(unique_urls)

    safe_print(f"\n📊 Search Phase Summary:")
    safe_print(f"   Time: {search_elapsed:.1f}s | Total: {len(all_properties)} | Unique: {len(unique_urls)}")

    if not unique_urls:
        safe_print("\n❌ No property URLs collected.")
        return pd.DataFrame()

    safe_print(f"\n🏠 Phase 2: Fetching details for {len(unique_urls)} properties (Parallel)...\n" + "-" * 60)

    start_time_details = time.time()
    all_condos = process_properties_parallel(unique_urls, max_detail_workers)
    detail_elapsed = time.time() - start_time_details

    safe_print("\n" + "=" * 60 + "\n✅ Scraping complete!")
    safe_print(f"\n⏱️  Performance Summary:")
    safe_print(f"   Phase 1 (Search):   {search_elapsed:.1f} seconds ({len(search_tasks)} pages)")
    safe_print(f"   Phase 2 (Details):  {detail_elapsed:.1f} seconds ({len(unique_urls)} properties)")
    safe_print(f"   Total time:         {(search_elapsed + detail_elapsed):.1f} seconds")

    safe_print(f"\n📈 Results:")
    safe_print(f"   Successfully scraped: {len(all_condos)}/{len(unique_urls)} properties")
    if (total_time := search_elapsed + detail_elapsed) > 0:
        safe_print(f"   Overall speed: {len(unique_urls) / total_time:.1f} properties/second")
        if detail_elapsed > 0:
            safe_print(f"   Detail phase speed: {len(unique_urls) / detail_elapsed:.1f} properties/second")

    return pd.DataFrame(all_condos) if all_condos else pd.DataFrame()

def scrape_condos_fully_parallel_improved(base_url, start_page=1, max_pages=5,
                                         max_search_workers=3, max_detail_workers=3,
                                         use_selenium_for_search=True):
    """Improved version with option to use Selenium for search pages"""
    safe_print(f"\n🚀 Starting IMPROVED PARALLEL scraping at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print(f"   Search Method: {'Selenium (JavaScript support)' if use_selenium_for_search else 'CloudScraper (faster but may miss content)'}")
    safe_print(f"   Search Workers: {max_search_workers}")
    safe_print(f"   Detail Workers: {max_detail_workers}")
    safe_print(f"   🛡️ Enhanced property detection enabled\n" + "=" * 60)

    search_tasks = [(page, base_url.replace('page=1', f'page={page}')) for page in range(start_page, start_page + max_pages)]

    safe_print(f"\n📚 Phase 1: Fetching {len(search_tasks)} search result pages...\n" + "-" * 60)
    all_properties = []

    start_time_search = time.time()

    # Choose fetching method
    fetch_function = fetch_search_page_selenium if use_selenium_for_search else fetch_search_page

    with ThreadPoolExecutor(max_workers=max_search_workers) as executor:
        futures = [executor.submit(fetch_function, task) for task in search_tasks]
        for future in as_completed(futures):
            try:
                page_num, properties = future.result()
                if properties:
                    all_properties.extend(properties)
                    # Log if we got the expected number
                    if len(properties) >= 50:
                        safe_print(f"   ✅ Page {page_num}: Got full set of {len(properties)} properties")
                    elif len(properties) >= 40:
                        safe_print(f"   ⚠️ Page {page_num}: Got {len(properties)} properties (slightly below expected)")
                    else:
                        safe_print(f"   ⚠️ Page {page_num}: Only got {len(properties)} properties (significantly below expected 52)")
            except Exception as e:
                safe_print(f"   ❌ Exception: {e}")

    search_elapsed = time.time() - start_time_search

    # Remove duplicates
    unique_urls = {prop['url']: prop for prop in all_properties}.values()
    unique_urls = list(unique_urls)

    safe_print(f"\n📊 Search Phase Summary:")
    safe_print(f"   Time: {search_elapsed:.1f}s")
    safe_print(f"   Total properties found: {len(all_properties)}")
    safe_print(f"   Unique properties: {len(unique_urls)}")
    safe_print(f"   Average per page: {len(all_properties) / len(search_tasks):.1f}")
    if len(all_properties) / len(search_tasks) < 45:
        safe_print(f"   ⚠️ Warning: Average properties per page is below 45 (expected ~52)")

    if not unique_urls:
        safe_print("\n❌ No property URLs collected.")
        return pd.DataFrame()

    safe_print(f"\n🏠 Phase 2: Fetching details for {len(unique_urls)} properties...\n" + "-" * 60)

    start_time_details = time.time()
    all_condos = process_properties_parallel(unique_urls, max_detail_workers)
    detail_elapsed = time.time() - start_time_details

    safe_print("\n" + "=" * 60 + "\n✅ Scraping complete!")
    safe_print(f"\n⏱️ Performance Summary:")
    safe_print(f"   Phase 1 (Search):   {search_elapsed:.1f} seconds ({len(search_tasks)} pages)")
    safe_print(f"   Phase 2 (Details):  {detail_elapsed:.1f} seconds ({len(unique_urls)} properties)")
    safe_print(f"   Total time:         {(search_elapsed + detail_elapsed):.1f} seconds")

    safe_print(f"\n📈 Results:")
    safe_print(f"   Successfully scraped: {len(all_condos)}/{len(unique_urls)} properties")
    safe_print(f"   Success rate: {len(all_condos)/len(unique_urls)*100:.1f}%")

    return pd.DataFrame(all_condos) if all_condos else pd.DataFrame()

def save_to_csv(df, filename=None):
    # Ensure data directory exists
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'toronto_condos_robust_{timestamp}.csv'

    # Save to data directory
    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False, encoding='utf-8')
    return filepath

def display_statistics(df):
    safe_print("\n📊 DATA STATISTICS\n" + "=" * 60)
    safe_print(f"Total properties: {len(df)}")
    
    if 'mls_number' in df.columns:
        has_mls = df['mls_number'].notna() & (df['mls_number'] != 'N/A')
        safe_print(f"Properties with MLS: {has_mls.sum()} ({has_mls.sum()/len(df)*100:.1f}%)")
    
    if 'price' in df.columns:
        try:
            df['price_numeric'] = pd.to_numeric(df['price'].str.replace(r'[$,]', '', regex=True), errors='coerce')
            valid_prices = df['price_numeric'].dropna()
            if not valid_prices.empty:
                safe_print("\nPrice Range:")
                safe_print(f"   Average: ${valid_prices.mean():,.0f} | Median: ${valid_prices.median():,.0f}")
                safe_print(f"   Min: ${valid_prices.min():,.0f} | Max: ${valid_prices.max():,.0f}")
        except:
            pass
    
    def count_non_empty_json(series):
        return series.apply(lambda x: len(json.loads(x)) > 0 if isinstance(x, str) and x.startswith('[') else False).sum()

    for col, name in [
        ('similar_listings', 'Similar Listings'),
        ('nearby_listings_for_sale', 'Nearby Sales Listings'),
        ('nearby_listings_for_rent', 'Nearby Rental Listings'),
        ('price_history_for_sale', 'Sale History'),
        ('price_history_for_lease', 'Lease History')
    ]:
        if col in df.columns:
            count = count_non_empty_json(df[col])
            safe_print(f"Properties with {name}: {count} ({count/len(df)*100:.1f}%)")

    if 'neighbourhood' in df.columns:
        top_neighborhoods = df['neighbourhood'].value_counts().head(5)
        if not top_neighborhoods.empty:
            safe_print(f"\nTop 5 Neighborhoods:")
            for neighborhood, count in top_neighborhoods.items():
                if neighborhood != 'N/A':
                    safe_print(f"   • {neighborhood}: {count} listings")

def check_system_resources():
    """Check system resources before starting"""
    try:
        import psutil
        safe_print("\n💻 System Resource Check:")
        safe_print(f"   CPU Cores: {psutil.cpu_count()}")
        safe_print(f"   RAM Available: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        safe_print(f"   Chrome processes running: {len([p for p in psutil.process_iter(['name']) if 'chrome' in p.info['name'].lower()])}")
    except ImportError:
        safe_print("\n💡 Tip: Install psutil for system resource monitoring (pip install psutil)")

def main():
    BASE_URL = "https://condos.ca/toronto?sublocality_id=14&mode=Sale&page=1"
    START_PAGE = 1
    MAX_PAGES = 30
    MAX_SEARCH_WORKERS = 10  # Reduced for Selenium search
    MAX_DETAIL_WORKERS = 16
    USE_SELENIUM_FOR_SEARCH = True  # Set to True for better detection

    safe_print("🏠 Toronto Condos ENHANCED Parallel Scraper\n" + "=" * 60)
    safe_print("🛡️ Features:")
    safe_print("   ✅ Selenium-based search page fetching (better JavaScript support)")
    safe_print("   ✅ Automatic retry for incomplete pages")
    safe_print("   ✅ Enhanced property detection")
    safe_print("   ✅ Automatic cleanup on exit")
    safe_print("   ✅ Thread-safe operations")

    check_system_resources()

    safe_print(f"\n⚙️ Configuration:")
    safe_print(f"   Pages to scrape: {MAX_PAGES}")
    safe_print(f"   Search method: {'Selenium' if USE_SELENIUM_FOR_SEARCH else 'CloudScraper'}")
    safe_print(f"   Search workers: {MAX_SEARCH_WORKERS}")
    safe_print(f"   Detail workers: {MAX_DETAIL_WORKERS}")
    safe_print(f"   Expected properties per page: ~52")
    safe_print("\n💡 Press Ctrl+C at any time for graceful shutdown\n")

    try:
        condos_df = scrape_condos_fully_parallel_improved(
            BASE_URL, START_PAGE, MAX_PAGES,
            MAX_SEARCH_WORKERS, MAX_DETAIL_WORKERS,
            USE_SELENIUM_FOR_SEARCH
        )

        if not condos_df.empty:
            display_statistics(condos_df)
            csv_filename = save_to_csv(condos_df)
            safe_print(f"\n✅ Data saved to: {csv_filename}")
        else:
            safe_print("\n❌ No data collected.")

    except KeyboardInterrupt:
        safe_print("\n\n🛑 Scraping interrupted by user")
    except Exception as e:
        safe_print(f"\n\n❌ Unexpected error: {str(e)}")
        if "--debug" in sys.argv:
            safe_print(f"Stack trace:\n{traceback.format_exc()}")
    finally:
        safe_print("\n🧹 Performing final cleanup check...")
        with driver_registry_lock:
            if active_drivers:
                safe_print(f"   ⚠️ Found {len(active_drivers)} active driver(s) - cleaning up...")
                cleanup_all_drivers()
            else:
                safe_print("   ✅ All drivers properly closed")
        safe_print("   👋 Scraper terminated cleanly")

if __name__ == "__main__":
    main()