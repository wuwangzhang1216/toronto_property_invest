#!/usr/bin/env python3
"""
Toronto Condo Investment Pipeline - Complete Version
Combines web scraping and investment analysis into a single automated workflow
Author: Toronto Real Estate Investment Analyzer
Version: 3.0
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import cloudscraper
import time
import random
from datetime import datetime
import json
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import sys
import os

warnings.filterwarnings('ignore')

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# Thread-safe print lock
print_lock = Lock()

def safe_print(message):
    """Thread-safe printing"""
    with print_lock:
        print(message)

# ============================================================================
# PART 1: WEB SCRAPING FUNCTIONS
# ============================================================================

def parse_listing_page(html_doc):
    """
    Parses the search results page to extract property URLs and MLS numbers.
    
    Args:
        html_doc (str): The HTML content of the listing page.
    
    Returns:
        list: A list of dictionaries containing URL and MLS for each property.
    """
    soup = BeautifulSoup(html_doc, 'html.parser')
    properties = []

    # Find the parent container that holds all the listing cards.
    list_container = soup.find('div', id='listRow')
    if not list_container:
        safe_print("Warning: Main listing container with id='listRow' not found.")
        return []

    # Find all individual listing cards
    listing_cards = list_container.find_all('div', class_=lambda c: c and 'styles___ListingPreview' in c)

    for card in listing_cards:
        # Get the property URL
        link_tag = card.find('a', href=True)
        if link_tag and link_tag['href']:
            href = link_tag['href']

            # Construct the full URL
            if not href.startswith('http'):
                href = f"https://condos.ca{href}"

            if '/unit-' in href:  # Valid listing URL
                # Extract MLS from the card text or URL
                card_text = card.get_text()
                
                # Look for Toronto MLS format (C/W/E/N/S followed by numbers)
                mls_match = re.search(r'\b([CWENS]\d{7,9})\b', card_text)
                if not mls_match:
                    # Try generic MLS pattern
                    mls_match = re.search(r'MLS[®#:\s]*([A-Z]?\d{7,9})', card_text)
                
                mls_number = mls_match.group(1) if mls_match else None
                
                properties.append({
                    'url': href,
                    'mls_number': mls_number
                })

    return properties


def extract_condo_details(html_content, url, mls_number=None):
    """
    Parses HTML content from an individual condo listing to extract detailed information.
    
    Args:
        html_content (str): The HTML content of the individual listing page.
        url (str): The URL of the property (for reference).
        mls_number (str): MLS number from the listing page.
    
    Returns:
        dict: A dictionary containing the extracted condo information.
    """
    if not html_content:
        return {}

    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Initialize with defaults - use MLS from listing page
    condo_data = {
        'url': url,
        'mls_number': mls_number if mls_number else 'N/A',
        'address': 'N/A',
        'building_name': 'N/A',
        'neighbourhood': 'N/A',
        'city': 'N/A',
        'price': 'N/A',
        'beds': 'N/A',
        'baths': 'N/A',
        'parking': 'N/A',
        'sqft': 'N/A',
        'property_type': 'N/A',
        'maintenance_fees': 'N/A',
        'taxes': 'N/A',
        'description': 'N/A',
        'price_history': 'N/A',
        'latest_leased_price': 'N/A',
        'latest_leased_date': 'N/A',
        'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    try:
        # Main Header Information
        title = soup.find('h1', class_=lambda c: c and '___Title-sc-' in c)
        if title:
            condo_data['address'] = title.text.strip()

        location_div = soup.find('div', class_=lambda c: c and '___Address-sc-' in c)
        if location_div:
            parts = [a.text.strip() for a in location_div.find_all('a')]
            if parts:
                condo_data['building_name'] = parts[0] if len(parts) > 0 else 'N/A'
                condo_data['neighbourhood'] = parts[1] if len(parts) > 1 else 'N/A'
                condo_data['city'] = parts[3] if len(parts) > 3 else 'N/A'
        
        price_div = soup.find('div', class_=lambda c: c and '___Price-sc-' in c)
        if price_div:
            condo_data['price'] = price_div.text.strip()

        # Unit Details
        details_div = soup.find('div', class_=lambda c: c and '___Details-sc-' in c)
        if details_div:
            items = details_div.find_all('div', recursive=False)
            for item in items:
                text = item.text.lower()
                if "bd" in text or "studio" in text:
                    condo_data['beds'] = item.text.strip()
                elif "ba" in text:
                    condo_data['baths'] = item.text.strip()
                elif "parking" in text:
                    condo_data['parking'] = item.text.strip()
                elif "sqft" in text:
                    condo_data['sqft'] = item.text.strip()
                elif "condo" in text or "townhouse" in text:
                    condo_data['property_type'] = item.text.strip()
        
        # Key Facts Section
        key_facts_section = soup.find('div', id='KeyFactsAndListingDetails')
        if key_facts_section:
            facts = key_facts_section.find_all('div', class_=lambda c: c and '___BlockContainer-sc-' in c)
            for fact in facts:
                title_div = fact.find('div', class_=lambda c: c and '___InfoRowTitle-sc-' in c)
                value_div = fact.find('div', class_=lambda c: c and '___InfoRowValue-sc-' in c)
                if title_div and value_div:
                    title = title_div.text.strip().lower()
                    value = value_div.text.strip()
                    if 'maintenance' in title:
                        condo_data['maintenance_fees'] = value
                    elif 'taxes' in title:
                        condo_data['taxes'] = value

        # Description
        description = soup.find('div', class_=lambda c: c and '___BodyHtml-sc-' in c)
        if description:
            condo_data['description'] = " ".join(description.text.strip().split())

        # Price History
        price_history = []
        history_container = soup.find('div', class_=lambda c: c and '___ArchiveListContainer-sc-' in c)
        if history_container:
            history_cards = history_container.find_all('a', class_=lambda c: c and '___ArchiveCard-sc-' in c)
            for card in history_cards[:5]:  # Limit to last 5 events
                event = {}
                date_tag = card.find('div', class_=lambda c: c and '___Date-sc-' in c)
                status_tag = card.find('div', class_=lambda c: c and '___Status-sc-' in c)

                if date_tag:
                    event['date'] = date_tag.text.strip()
                
                if status_tag:
                    status = status_tag.find('span').text.strip() if status_tag.find('span') else 'N/A'
                    price = status_tag.find('div').text.strip() if status_tag.find('div') else 'N/A'
                    event['status'] = status
                    event['price'] = price
                
                if event:
                    price_history.append(event)
        
        # Store price history as JSON string
        if price_history:
            condo_data['price_history'] = json.dumps(price_history)
            
            # Extract latest leased price
            latest_lease = next((item for item in price_history if 'Leased' in item.get('status', '')), None)
            if latest_lease:
                condo_data['latest_leased_price'] = latest_lease.get('price', 'N/A')
                condo_data['latest_leased_date'] = latest_lease.get('date', 'N/A')

    except Exception as e:
        safe_print(f"      Error parsing {url}: {e}")
    
    return condo_data


def fetch_page(url, delay_range=(0.5, 1.5)):
    """
    Fetches HTML content using cloudscraper.
    """
    time.sleep(random.uniform(*delay_range))
    scraper = cloudscraper.create_scraper()
    try:
        response = scraper.get(url, timeout=60)
        if response.status_code == 200:
            return response.text
        else:
            safe_print(f"      Failed to fetch {url} with status code: {response.status_code}")
    except Exception as e:
        safe_print(f"      Failed to fetch {url}: {e}")
    return None


def fetch_search_page(args):
    """
    Worker function to fetch and parse a search results page.
    
    Args:
        args: tuple of (page_num, url)
    
    Returns:
        tuple: (page_num, list of property dictionaries with url and mls)
    """
    page_num, url = args
    safe_print(f"📄 Fetching search page {page_num}...")
    
    html_content = fetch_page(url, delay_range=(0.2, 0.5))
    if html_content:
        properties = parse_listing_page(html_content)
        if properties:
            with_mls = sum(1 for p in properties if p.get('mls_number'))
            safe_print(f"   ✅ Page {page_num}: Found {len(properties)} properties ({with_mls} with MLS)")
            return page_num, properties
        else:
            safe_print(f"   ⚠️ Page {page_num}: No properties found")
            return page_num, []
    else:
        safe_print(f"   ❌ Page {page_num}: Failed to fetch")
        return page_num, []


def fetch_property_details(args):
    """
    Worker function to fetch and extract property details.
    """
    property_dict, index, total = args
    property_url = property_dict['url']
    mls_number = property_dict.get('mls_number')
    
    safe_print(f"  [{index:>3}/{total}] Processing: {property_url.split('/')[-1][:40]}...")
    html = fetch_page(property_url)
    if html:
        details = extract_condo_details(html, property_url, mls_number)
        if details and details.get('address') != 'N/A':
            safe_print(f"  [{index:>3}/{total}] ✅ Success | {details.get('address', 'Unknown')[:40]} - {details.get('price', 'N/A')}")
            return details
        else:
            safe_print(f"  [{index:>3}/{total}] ⚠️ Failed to extract details from page.")
    else:
        safe_print(f"  [{index:>3}/{total}] ❌ Failed to fetch page content.")
    return None


def scrape_condos_fully_parallel(base_url, start_page=1, max_pages=5, 
                                  max_search_workers=5, max_detail_workers=10):
    """
    Main function to scrape condo information using full parallelization.
    Both search pages and property details are fetched in parallel.
    """
    print(f"\n🚀 Starting FULLY PARALLEL scraping at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Search page workers: {max_search_workers} threads")
    print(f"   Property detail workers: {max_detail_workers} threads")
    print("=" * 60)
    
    # Step 1: Prepare URLs for all search pages
    search_tasks = []
    for page_num in range(start_page, start_page + max_pages):
        url = base_url.replace('page=2', f'page={page_num}')
        search_tasks.append((page_num, url))
    
    # Step 2: Fetch all search pages in parallel
    print(f"\n📚 Phase 1: Fetching {len(search_tasks)} search result pages in parallel...")
    print("-" * 60)
    
    all_properties = []
    pages_data = {}
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_search_workers) as executor:
        future_to_page = {executor.submit(fetch_search_page, task): task[0] for task in search_tasks}
        
        for future in as_completed(future_to_page):
            try:
                page_num, properties = future.result()
                if properties:
                    pages_data[page_num] = properties
                    all_properties.extend(properties)
            except Exception as e:
                safe_print(f"   ❌ Exception in search page worker: {e}")
    
    search_elapsed = time.time() - start_time
    
    print(f"\n📊 Search Phase Summary:")
    print(f"   Time taken: {search_elapsed:.1f} seconds")
    print(f"   Pages successfully fetched: {len(pages_data)}/{len(search_tasks)}")
    print(f"   Total property URLs collected: {len(all_properties)}")
    
    if not all_properties:
        print("\n❌ No property URLs were collected. Exiting.")
        return pd.DataFrame()

    # Remove duplicates based on URL
    unique_urls = []
    seen_urls = set()
    for prop in all_properties:
        if prop['url'] not in seen_urls:
            unique_urls.append(prop)
            seen_urls.add(prop['url'])
    
    print(f"   Unique properties to scrape: {len(unique_urls)}")
    
    # Step 3: Fetch all property details in parallel
    print(f"\n🏠 Phase 2: Fetching {len(unique_urls)} property details in parallel...")
    print("-" * 60)
    
    task_args = [(prop, i + 1, len(unique_urls)) for i, prop in enumerate(unique_urls)]
    all_condos = []
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_detail_workers) as executor:
        future_to_url = {executor.submit(fetch_property_details, args): args[0] for args in task_args}
        
        for future in as_completed(future_to_url):
            try:
                result = future.result()
                if result:
                    all_condos.append(result)
            except Exception as e:
                safe_print(f"   ❌ Exception in property detail worker: {e}")

    detail_elapsed = time.time() - start_time
    
    # Final summary
    total_elapsed = search_elapsed + detail_elapsed
    
    print("\n" + "=" * 60)
    print(f"✅ Scraping complete!")
    print(f"\n⏱️  Performance Summary:")
    print(f"   Phase 1 (Search pages):    {search_elapsed:.1f} seconds")
    print(f"   Phase 2 (Property details): {detail_elapsed:.1f} seconds")
    print(f"   Total time:                 {total_elapsed:.1f} seconds")
    print(f"\n📈 Results:")
    print(f"   Properties successfully scraped: {len(all_condos)}/{len(unique_urls)}")
    if total_elapsed > 0:
        print(f"   Overall speed: {len(unique_urls) / total_elapsed:.1f} properties/second")
    
    if all_condos:
        df = pd.DataFrame(all_condos)
        return df
    
    return pd.DataFrame()


# ============================================================================
# PART 2: INVESTMENT ANALYSIS CLASS
# ============================================================================

class CondoInvestmentAnalyzer:
    """
    Analyzes condo investments based on rental yield (租售比) and net income after expenses.
    Outputs a comprehensive HTML report with interactive charts and tables.
    """
    
    def __init__(self, df, mortgage_rate=3.5, property_tax_rate=0.5, down_payment_percent=20, min_price=200000):
        """
        Initialize the analyzer with data and parameters.
        
        Args:
            df: DataFrame with condo data
            mortgage_rate: Annual mortgage interest rate (default 3.5%)
            property_tax_rate: Annual property tax rate (default 0.5%)
            down_payment_percent: Down payment percentage (default 20%)
            min_price: Minimum price to filter out parking/rentals (default $200,000)
        """
        self.df = df
        self.mortgage_rate = mortgage_rate / 100  # Convert to decimal
        self.property_tax_rate = property_tax_rate / 100  # Convert to decimal
        self.down_payment_percent = down_payment_percent / 100  # Convert to decimal
        self.min_price = min_price
        self.rental_estimates = self._load_rental_estimates()
        
        # Initial filtering for minimum price
        initial_count = len(self.df)
        self.df = self._filter_minimum_price()
        filtered_count = len(self.df)
        
        print(f"\n📊 Loaded {initial_count} properties for analysis")
        print(f"   Filtered to {filtered_count} properties (price ≥ ${min_price:,.0f})")
        print(f"   Mortgage Rate: {mortgage_rate}% | Property Tax: {property_tax_rate}% | Down Payment: {down_payment_percent}%")
    
    def _filter_minimum_price(self):
        """Filter out properties below minimum price (parking spots, rentals)."""
        # Clean prices first
        self.df['price_numeric'] = self.df['price'].apply(self.clean_price)
        
        # Filter by minimum price and valid prices
        filtered_df = self.df[
            (self.df['price_numeric'].notna()) & 
            (self.df['price_numeric'] >= self.min_price)
        ].copy()
        
        return filtered_df
    
    def _load_rental_estimates(self):
        """
        Load Toronto rental market estimates (2024-2025 market rates).
        Returns dict with rental estimates by bedroom count.
        """
        return {
            'Studio': 1900,
            '1': 2400,
            '1+1': 2600,
            '2': 3200,
            '2+1': 3400,
            '3': 4200,
            '3+1': 4500,
            '4': 5500,
        }
    
    def clean_price(self, price_str):
        """Convert price string to float."""
        if pd.isna(price_str) or price_str == 'N/A':
            return np.nan
        
        price_str = str(price_str).replace('$', '').replace(',', '').strip()
        
        if not price_str or not price_str[0].isdigit():
            return np.nan
        
        try:
            return float(price_str)
        except:
            return np.nan
    
    def clean_maintenance_fee(self, fee_str):
        """Convert maintenance fee string to monthly float."""
        if pd.isna(fee_str) or fee_str == 'N/A':
            return 0
        
        fee_str = str(fee_str).replace('$', '').replace(',', '').strip()
        
        import re
        match = re.search(r'(\d+(?:\.\d+)?)', fee_str)
        if match:
            return float(match.group(1))
        return 0
    
    def parse_bedroom_info(self, beds_str):
        """Parse bedroom information from various formats."""
        if pd.isna(beds_str) or beds_str == 'N/A':
            return None
        
        beds_str = str(beds_str).lower()
        
        # Handle "Studio" case - must check before bed pattern
        if 'studio' in beds_str:
            return 'Studio'
        
        import re
        # Look for pattern like "1+1 bed" or "2 bed" after studio check
        match = re.search(r'(\d+(?:\+\d+)?)\s*bed', beds_str)
        if match:
            return match.group(1)
        
        return None
    
    def parse_bath_info(self, bath_str):
        """Parse bathroom count from various formats."""
        if pd.isna(bath_str) or bath_str == 'N/A':
            return None
        
        bath_str = str(bath_str).lower()
        
        import re
        match = re.search(r'(\d+)\s*bath', bath_str)
        if match:
            return int(match.group(1))
        
        return None
    
    def parse_parking_info(self, parking_str):
        """Parse parking spaces from various formats."""
        if pd.isna(parking_str) or parking_str == 'N/A':
            return 0
        
        parking_str = str(parking_str).lower()
        
        import re
        match = re.search(r'(\d+)\s*parking', parking_str)
        if match:
            return int(match.group(1))
        
        return 0
    
    def parse_sqft_info(self, sqft_str):
        """Parse square footage from various formats."""
        if pd.isna(sqft_str) or sqft_str == 'N/A':
            return None
        
        sqft_str = str(sqft_str).lower()
        
        import re
        match = re.search(r'(\d+)\s*sqft', sqft_str)
        if match:
            return f"{match.group(1)} sqft"
        
        return None
    
    def estimate_rental_income(self, row):
        """
        Estimate monthly rental income based on property characteristics.
        """
        # Use the cleaned beds column
        bed_info = row.get('beds', None)
        
        if bed_info and bed_info in self.rental_estimates:
            base_rent = self.rental_estimates[bed_info]
        else:
            # Default to 1-bedroom estimate if no bedroom info
            base_rent = self.rental_estimates.get('1', 2400)
        
        # Adjust for square footage if available
        sqft = row.get('sqft', None)
        if sqft and sqft != 'N/A':
            import re
            sqft_match = re.search(r'(\d+)', str(sqft))
            if sqft_match:
                sqft_num = int(sqft_match.group(1))
                if sqft_num > 0:
                    # Weight: 60% bedroom-based, 40% sqft-based
                    sqft_based_rent = sqft_num * 3.75
                    base_rent = (base_rent * 0.6) + (sqft_based_rent * 0.4)
        
        # Neighborhood adjustments for premium areas
        neighbourhood = str(row.get('neighbourhood', '')).lower()
        premium_areas = ['yorkville', 'king west', 'queen west', 'liberty village', 
                       'distillery', 'harbourfront', 'financial district', 'bay street',
                       'st. lawrence', 'waterfront']
        if any(area in neighbourhood for area in premium_areas):
            base_rent *= 1.15
        
        return base_rent
    
    def calculate_investment_metrics(self):
        """
        Calculate all investment metrics including rental yield and net income.
        """
        print("\n🔍 Calculating investment metrics...")
        
        self.df = self.df[self.df['price_numeric'] > 0].copy()
        
        # Process beds column first
        self.df['beds_cleaned'] = self.df['beds'].apply(lambda x: 
            self.parse_bedroom_info(x) if pd.notna(x) and x != 'N/A' else None
        )
        
        # If beds_cleaned is None, try to extract from baths column
        self.df['beds_cleaned'] = self.df.apply(lambda row: 
            row['beds_cleaned'] if row['beds_cleaned'] 
            else self.parse_bedroom_info(row.get('baths', 'N/A')), 
            axis=1
        )
        
        # Process sqft - check both columns for combined string
        self.df['sqft_cleaned'] = self.df.apply(lambda row:
            self.parse_sqft_info(row.get('sqft', 'N/A')) or 
            self.parse_sqft_info(row.get('beds', 'N/A')) or 
            self.parse_sqft_info(row.get('baths', 'N/A')),
            axis=1
        )
        
        # Process bath count
        self.df['bath_count'] = self.df.apply(lambda row:
            self.parse_bath_info(row.get('baths', 'N/A')) or
            self.parse_bath_info(row.get('beds', 'N/A')),
            axis=1
        )
        
        # Process parking
        self.df['parking_count'] = self.df.apply(lambda row:
            self.parse_parking_info(row.get('parking', 'N/A')) or
            self.parse_parking_info(row.get('beds', 'N/A')) or
            self.parse_parking_info(row.get('baths', 'N/A')),
            axis=1
        )
        
        # Update the main columns with cleaned values
        self.df['beds'] = self.df['beds_cleaned']
        self.df['sqft'] = self.df['sqft_cleaned']
        
        # Clean maintenance fees
        self.df['maintenance_fee_monthly'] = self.df['maintenance_fees'].apply(self.clean_maintenance_fee)
        
        # Estimate monthly rental income
        self.df['estimated_monthly_rent'] = self.df.apply(self.estimate_rental_income, axis=1)
        self.df['annual_rental_income'] = self.df['estimated_monthly_rent'] * 12
        
        # Calculate gross rental yield
        self.df['gross_rental_yield'] = (self.df['annual_rental_income'] / self.df['price_numeric']) * 100
        
        # Calculate annual expenses
        self.df['annual_mortgage_interest'] = self.df['price_numeric'] * self.mortgage_rate
        self.df['annual_property_tax'] = self.df['price_numeric'] * self.property_tax_rate
        self.df['annual_maintenance'] = self.df['maintenance_fee_monthly'] * 12
        
        # Calculate total expenses
        self.df['total_annual_expenses'] = (
            self.df['annual_mortgage_interest'] + 
            self.df['annual_property_tax'] + 
            self.df['annual_maintenance']
        )
        
        # Calculate net rental yield
        self.df['net_annual_income'] = self.df['annual_rental_income'] - self.df['total_annual_expenses']
        self.df['net_rental_yield'] = (self.df['net_annual_income'] / self.df['price_numeric']) * 100
        
        # Calculate cash-on-cash return
        self.df['down_payment'] = self.df['price_numeric'] * self.down_payment_percent
        self.df['cash_on_cash_return'] = (self.df['net_annual_income'] / self.df['down_payment']) * 100
        
        # Calculate expense ratio
        self.df['expense_ratio'] = (self.df['total_annual_expenses'] / self.df['annual_rental_income']) * 100
        
        # Create investment score
        self.df['investment_score'] = (
            self.df['net_rental_yield'] * 0.4 +
            self.df['gross_rental_yield'] * 0.2 +
            self.df['cash_on_cash_return'] * 0.2 +
            (100 - self.df['expense_ratio'].clip(upper=100)) * 0.2
        )
        
        print(f"   ✅ Metrics calculated for {len(self.df)} properties")
        
        # Debug info for parsed data
        valid_beds = self.df[self.df['beds'].notna()]['beds'].count()
        valid_sqft = self.df[self.df['sqft'].notna()]['sqft'].count()
        print(f"   • Properties with valid bedroom info: {valid_beds}")
        print(f"   • Properties with valid sqft info: {valid_sqft}")
        
        return self.df
    
    def filter_good_investments(self, min_gross_yield=7, min_net_yield=3):
        """
        Filter properties that meet investment criteria.
        """
        print(f"\n🎯 Filtering for properties with:")
        print(f"   • Gross rental yield ≥ {min_gross_yield}%")
        print(f"   • Net rental yield ≥ {min_net_yield}%")
        print(f"   • Based on {int(self.down_payment_percent * 100)}% down payment")
        
        good_investments = self.df[
            (self.df['gross_rental_yield'] >= min_gross_yield) &
            (self.df['net_rental_yield'] >= min_net_yield)
        ].copy()
        
        good_investments = good_investments.sort_values('investment_score', ascending=False)
        
        print(f"\n💎 Found {len(good_investments)} properties meeting criteria")
        
        return good_investments
    
    def generate_html_report(self, good_investments, filename=None):
        """
        Generate a comprehensive HTML report with charts and tables.
        """
        print("\n📝 Generating HTML report...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if filename is None:
            filename = f'condo_investment_report_{timestamp}.html'
        
        # Prepare data for charts
        all_properties = self.df.copy()
        top_20 = all_properties.nlargest(20, 'investment_score')
        
        print(f"   • Processing {len(all_properties)} properties for visualization")
        
        # Calculate statistics
        stats = {
            'total_properties': len(all_properties),
            'good_investments': len(good_investments),
            'good_percentage': (len(good_investments) / len(all_properties) * 100) if len(all_properties) > 0 else 0,
            'avg_price_all': all_properties['price_numeric'].mean(),
            'median_price_all': all_properties['price_numeric'].median(),
            'avg_price_good': good_investments['price_numeric'].mean() if len(good_investments) > 0 else 0,
            'median_price_good': good_investments['price_numeric'].median() if len(good_investments) > 0 else 0,
            'avg_gross_yield_all': all_properties['gross_rental_yield'].mean(),
            'avg_net_yield_all': all_properties['net_rental_yield'].mean(),
            'avg_gross_yield_good': good_investments['gross_rental_yield'].mean() if len(good_investments) > 0 else 0,
            'avg_net_yield_good': good_investments['net_rental_yield'].mean() if len(good_investments) > 0 else 0,
            'avg_cash_return_all': all_properties['cash_on_cash_return'].mean(),
            'avg_cash_return_good': good_investments['cash_on_cash_return'].mean() if len(good_investments) > 0 else 0,
        }
        
        # Prepare chart data with error handling
        gross_yield_clean = all_properties['gross_rental_yield'].dropna()
        net_yield_clean = all_properties['net_rental_yield'].dropna()
        cash_return_clean = all_properties['cash_on_cash_return'].dropna()
        score_clean = all_properties['investment_score'].dropna()
        
        # Create histograms with fallback for empty data
        gross_yield_hist = np.histogram(gross_yield_clean, bins=min(30, max(5, len(gross_yield_clean)//10))) if len(gross_yield_clean) > 0 else (np.array([0]), np.array([0, 1]))
        net_yield_hist = np.histogram(net_yield_clean, bins=min(30, max(5, len(net_yield_clean)//10))) if len(net_yield_clean) > 0 else (np.array([0]), np.array([0, 1]))
        cash_return_hist = np.histogram(cash_return_clean, bins=min(30, max(5, len(cash_return_clean)//10))) if len(cash_return_clean) > 0 else (np.array([0]), np.array([0, 1]))
        score_hist = np.histogram(score_clean, bins=min(20, max(5, len(score_clean)//10))) if len(score_clean) > 0 else (np.array([0]), np.array([0, 1]))
        
        # Top neighborhoods (exclude N/A and empty strings)
        valid_neighborhoods = all_properties[
            all_properties['neighbourhood'].notna() & 
            (all_properties['neighbourhood'] != 'N/A') & 
            (all_properties['neighbourhood'] != '') &
            (all_properties['neighbourhood'].str.strip() != '')
        ]['neighbourhood'].str.strip()
        
        neighborhood_counts = valid_neighborhoods.value_counts().head(10)
        
        if len(neighborhood_counts) > 0:
            neighborhood_labels = [str(x) for x in neighborhood_counts.index.tolist()]
            neighborhood_values = [int(x) for x in neighborhood_counts.values.tolist()]
            print(f"   • Found {len(neighborhood_labels)} neighborhoods with data")
        else:
            neighborhood_labels = []
            neighborhood_values = []
            print("   • Warning: No valid neighborhood data found")
        
        # Generate the full HTML content
        html_content = self._generate_full_html_report(
            stats, all_properties, top_20, good_investments,
            gross_yield_hist, net_yield_hist, cash_return_hist, score_hist,
            neighborhood_labels, neighborhood_values
        )
        
        # Save HTML file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✅ HTML report generated: {filename}")
        print(f"   • Total charts: 6")
        print(f"   • Properties analyzed: {len(all_properties)}")
        print(f"   • Good investments found: {len(good_investments)}")
        
        return filename
    
    def _generate_full_html_report(self, stats, all_properties, top_20, good_investments,
                                   gross_yield_hist, net_yield_hist, cash_return_hist, score_hist,
                                   neighborhood_labels, neighborhood_values):
        """Generate the complete HTML report with all charts and tables."""
        
        # Prepare JavaScript data for properties
        top_properties_data = []
        for idx, (prop_idx, row) in enumerate(top_20.iterrows(), 1):
            top_prop_id = f"top_prop_{idx}"
            monthly_mortgage = row['annual_mortgage_interest'] / 12
            monthly_tax = row['annual_property_tax'] / 12
            monthly_maintenance = row['maintenance_fee_monthly']
            monthly_total_expense = monthly_mortgage + monthly_tax + monthly_maintenance
            monthly_net_income = row['estimated_monthly_rent'] - monthly_total_expense
            
            top_prop_data = {
                'id': top_prop_id,
                'address': str(row['address'])[:35] if pd.notna(row['address']) else 'N/A',
                'url': row['url'] if pd.notna(row['url']) else '#',
                'neighbourhood': row['neighbourhood'] if pd.notna(row['neighbourhood']) else 'N/A',
                'beds': row['beds'] if pd.notna(row['beds']) else 'N/A',
                'sqft': row['sqft'] if pd.notna(row['sqft']) else 'N/A',
                'price': float(row['price_numeric']),
                'downPayment': float(row['down_payment']),
                'monthlyRent': float(row['estimated_monthly_rent']),
                'annualRent': float(row['annual_rental_income']),
                'monthlyMortgage': float(monthly_mortgage),
                'monthlyTax': float(monthly_tax),
                'monthlyMaintenance': float(monthly_maintenance),
                'monthlyTotalExpense': float(monthly_total_expense),
                'monthlyNetIncome': float(monthly_net_income),
                'annualNetIncome': float(row['net_annual_income']),
                'grossYield': float(row['gross_rental_yield']),
                'netYield': float(row['net_rental_yield']),
                'cashReturn': float(row['cash_on_cash_return']),
                'expenseRatio': float(row['expense_ratio']),
                'investmentScore': float(row['investment_score'])
            }
            top_properties_data.append(top_prop_data)
        
        properties_data = []
        if len(good_investments) > 0:
            for idx, (prop_idx, row) in enumerate(good_investments.head(50).iterrows(), 1):
                prop_id = f"prop_{idx}"
                monthly_mortgage = row['annual_mortgage_interest'] / 12
                monthly_tax = row['annual_property_tax'] / 12
                monthly_maintenance = row['maintenance_fee_monthly']
                monthly_total_expense = monthly_mortgage + monthly_tax + monthly_maintenance
                monthly_net_income = row['estimated_monthly_rent'] - monthly_total_expense
                
                prop_data = {
                    'id': prop_id,
                    'address': str(row['address'])[:35] if pd.notna(row['address']) else 'N/A',
                    'url': row['url'] if pd.notna(row['url']) else '#',
                    'price': float(row['price_numeric']),
                    'downPayment': float(row['down_payment']),
                    'monthlyRent': float(row['estimated_monthly_rent']),
                    'annualRent': float(row['annual_rental_income']),
                    'monthlyMortgage': float(monthly_mortgage),
                    'monthlyTax': float(monthly_tax),
                    'monthlyMaintenance': float(monthly_maintenance),
                    'monthlyTotalExpense': float(monthly_total_expense),
                    'monthlyNetIncome': float(monthly_net_income),
                    'annualNetIncome': float(row['net_annual_income']),
                    'grossYield': float(row['gross_rental_yield']),
                    'netYield': float(row['net_rental_yield']),
                    'cashReturn': float(row['cash_on_cash_return']),
                    'expenseRatio': float(row['expense_ratio'])
                }
                properties_data.append(prop_data)
        
        # Prepare chart data for JavaScript
        gross_yield_data = {
            'labels': [f"{x:.1f}" for x in gross_yield_hist[1][:-1]][:30],
            'values': gross_yield_hist[0].tolist()[:30]
        }
        
        net_yield_data = {
            'labels': [f"{x:.1f}" for x in net_yield_hist[1][:-1]][:30],
            'values': net_yield_hist[0].tolist()[:30]
        }
        
        cash_return_data = {
            'labels': [f"{x:.1f}" for x in cash_return_hist[1][:-1]][:30],
            'values': cash_return_hist[0].tolist()[:30]
        }
        
        score_data = {
            'labels': [f"{x:.1f}" for x in score_hist[1][:-1]][:20],
            'values': score_hist[0].tolist()[:20]
        }
        
        # Scatter plot data
        scatter_sample = all_properties.sample(min(500, len(all_properties)))
        scatter_points = []
        for _, row in scatter_sample.iterrows():
            if pd.notna(row['price_numeric']) and pd.notna(row['net_rental_yield']):
                scatter_points.append({
                    'x': float(row['price_numeric'] / 1000),
                    'y': float(row['net_rental_yield'])
                })
        
        # Build top properties table HTML
        top_properties_html = ""
        for idx, (prop_idx, row) in enumerate(top_20.iterrows(), 1):
            score_class = 'score-excellent' if row['investment_score'] > 20 else 'score-good' if row['investment_score'] > 15 else 'score-fair'
            badge_class = 'badge-excellent' if row['net_rental_yield'] > 4 else 'badge-good' if row['net_rental_yield'] > 3 else 'badge-fair'
            top_prop_id = f"top_prop_{idx}"
            
            top_properties_html += f"""
                        <tr class="clickable" data-prop-id="{top_prop_id}" onclick="toggleBreakdown('{top_prop_id}', event)">
                            <td>{idx}<span class="expand-indicator">▶</span></td>
                            <td>{str(row['address'])[:30] if pd.notna(row['address']) else 'N/A'}</td>
                            <td>{row['neighbourhood'] if pd.notna(row['neighbourhood']) else 'N/A'}</td>
                            <td>${row['price_numeric']:,.0f}</td>
                            <td>{row['beds'] if pd.notna(row['beds']) else 'N/A'}</td>
                            <td><span class="badge {badge_class}">{row['gross_rental_yield']:.2f}%</span></td>
                            <td><span class="badge {badge_class}">{row['net_rental_yield']:.2f}%</span></td>
                            <td class="{score_class}">{row['investment_score']:.1f}</td>
                            <td><a href="{row['url'] if pd.notna(row['url']) else '#'}" target="_blank" onclick="event.stopPropagation()">View</a></td>
                        </tr>
                        <tr id="{top_prop_id}_breakdown" style="display: none;">
                            <td colspan="9" style="padding: 0;">
                                <div class="calc-breakdown" id="{top_prop_id}_content">
                                    <!-- Content will be populated by JavaScript -->
                                </div>
                            </td>
                        </tr>
"""
        
        # Build good investments table HTML
        good_investments_html = ""
        if len(good_investments) > 0:
            for idx, (prop_idx, row) in enumerate(good_investments.head(50).iterrows(), 1):
                prop_id = f"prop_{idx}"
                good_investments_html += f"""
                        <tr class="clickable" data-prop-id="{prop_id}" onclick="toggleBreakdown('{prop_id}', event)">
                            <td>{idx}<span class="expand-indicator">▶</span></td>
                            <td>{str(row['address'])[:35] if pd.notna(row['address']) else 'N/A'}</td>
                            <td>${row['price_numeric']:,.0f}</td>
                            <td>{row['beds'] if pd.notna(row['beds']) else 'N/A'}</td>
                            <td>{row['gross_rental_yield']:.2f}%</td>
                            <td>{row['net_rental_yield']:.2f}%</td>
                            <td>{row['cash_on_cash_return']:.1f}%</td>
                            <td><a href="{row['url'] if pd.notna(row['url']) else '#'}" target="_blank" onclick="event.stopPropagation()">View</a></td>
                        </tr>
                        <tr id="{prop_id}_breakdown" style="display: none;">
                            <td colspan="8" style="padding: 0;">
                                <div class="calc-breakdown" id="{prop_id}_content">
                                    <!-- Content will be populated by JavaScript -->
                                </div>
                            </td>
                        </tr>
"""
        
        # Generate complete HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Toronto Condo Investment Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #0f0f0f;
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: #1a1a1a;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
            border-bottom: 1px solid #333;
            color: #e0e0e0;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
            font-weight: 600;
            letter-spacing: -0.5px;
        }}
        
        .header .subtitle {{
            font-size: 1em;
            color: #888;
        }}
        
        .header .params {{
            margin-top: 20px;
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
        }}
        
        .header .param {{
            background: rgba(255,255,255,0.05);
            padding: 8px 16px;
            border-radius: 6px;
            border: 1px solid rgba(255,255,255,0.1);
            font-size: 0.9em;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .summary-card {{
            background: #242424;
            border: 1px solid #333;
            color: #e0e0e0;
            padding: 25px;
            border-radius: 8px;
        }}
        
        .summary-card.highlight {{
            background: #1e2a1e;
            border-color: #2e7d32;
        }}
        
        .summary-card.warning {{
            background: #2a1e1e;
            border-color: #c62828;
        }}
        
        .summary-card h3 {{
            font-size: 0.85em;
            color: #888;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .summary-card .value {{
            font-size: 1.8em;
            font-weight: 600;
            color: #fff;
        }}
        
        .summary-card .subvalue {{
            font-size: 0.9em;
            color: #888;
            margin-top: 5px;
        }}
        
        .chart-section {{
            margin-bottom: 40px;
        }}
        
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }}
        
        .chart-container {{
            background: #242424;
            border: 1px solid #333;
            padding: 25px;
            border-radius: 8px;
            height: 380px;
        }}
        
        .chart-container h3 {{
            margin-bottom: 20px;
            color: #e0e0e0;
            font-size: 1.1em;
            font-weight: 500;
        }}
        
        .table-section {{
            margin-top: 40px;
        }}
        
        .table-section h2 {{
            color: #e0e0e0;
            margin-bottom: 20px;
            font-size: 1.5em;
            font-weight: 500;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }}
        
        .property-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: #1e1e1e;
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .property-table thead {{
            background: #242424;
        }}
        
        .property-table th {{
            padding: 14px;
            text-align: left;
            font-weight: 500;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #888;
            border-bottom: 1px solid #333;
        }}
        
        .property-table td {{
            padding: 12px 14px;
            border-bottom: 1px solid #2a2a2a;
            font-size: 0.9em;
            color: #d0d0d0;
        }}
        
        .property-table tbody tr:hover {{
            background: #252525;
        }}
        
        .property-table tbody tr.clickable {{
            cursor: pointer;
        }}
        
        .property-table tbody tr.clickable:hover {{
            background: #2a2a2a;
        }}
        
        .property-table .score-excellent {{
            color: #4caf50;
            font-weight: 600;
        }}
        
        .property-table .score-good {{
            color: #2196f3;
            font-weight: 600;
        }}
        
        .property-table .score-fair {{
            color: #ff9800;
            font-weight: 600;
        }}
        
        .property-table a {{
            color: #64b5f6;
            text-decoration: none;
        }}
        
        .property-table a:hover {{
            text-decoration: underline;
        }}
        
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        
        .badge-excellent {{
            background: rgba(76, 175, 80, 0.15);
            color: #4caf50;
            border: 1px solid rgba(76, 175, 80, 0.3);
        }}
        
        .badge-good {{
            background: rgba(33, 150, 243, 0.15);
            color: #2196f3;
            border: 1px solid rgba(33, 150, 243, 0.3);
        }}
        
        .badge-fair {{
            background: rgba(255, 152, 0, 0.15);
            color: #ff9800;
            border: 1px solid rgba(255, 152, 0, 0.3);
        }}
        
        /* Calculation Breakdown Styles */
        .calc-breakdown {{
            display: none;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            margin: 15px 0;
            padding: 20px;
        }}
        
        .calc-breakdown.show {{
            display: block;
            animation: slideDown 0.3s ease-out;
        }}
        
        @keyframes slideDown {{
            from {{
                opacity: 0;
                transform: translateY(-10px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        .calc-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        
        .calc-section {{
            background: #242424;
            border: 1px solid #2a2a2a;
            border-radius: 6px;
            padding: 15px;
        }}
        
        .calc-section h4 {{
            color: #888;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #2a2a2a;
        }}
        
        .calc-line {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        
        .calc-line:last-child {{
            border-bottom: none;
        }}
        
        .calc-line.total {{
            margin-top: 8px;
            padding-top: 12px;
            border-top: 2px solid #333;
            font-weight: 600;
        }}
        
        .calc-line.highlight {{
            background: rgba(76, 175, 80, 0.1);
            padding: 8px;
            border-radius: 4px;
            margin: 8px 0;
        }}
        
        .calc-label {{
            color: #aaa;
            font-size: 0.9em;
        }}
        
        .calc-value {{
            color: #fff;
            font-weight: 500;
        }}
        
        .calc-value.positive {{
            color: #4caf50;
        }}
        
        .calc-value.negative {{
            color: #f44336;
        }}
        
        .calc-value.warning {{
            color: #ff9800;
        }}
        
        .expand-indicator {{
            display: inline-block;
            margin-left: 5px;
            color: #666;
            font-size: 0.8em;
            transition: transform 0.2s;
        }}
        
        tr.expanded .expand-indicator {{
            transform: rotate(90deg);
        }}
        
        .criteria-met {{
            display: inline-block;
            background: rgba(76, 175, 80, 0.15);
            color: #4caf50;
            border: 1px solid rgba(76, 175, 80, 0.3);
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 0.85em;
            margin-left: 10px;
        }}
        
        .criteria-not-met {{
            display: inline-block;
            background: rgba(244, 67, 54, 0.15);
            color: #f44336;
            border: 1px solid rgba(244, 67, 54, 0.3);
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 0.85em;
            margin-left: 10px;
        }}
        
        .footer {{
            background: #1e1e1e;
            padding: 30px;
            text-align: center;
            color: #666;
            border-top: 1px solid #333;
        }}
        
        .legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            color: #888;
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }}
        
        .click-hint {{
            color: #666;
            font-size: 0.85em;
            font-style: italic;
            margin-bottom: 10px;
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.5em;
            }}
            
            .chart-grid {{
                grid-template-columns: 1fr;
            }}
            
            .calc-grid {{
                grid-template-columns: 1fr;
            }}
            
            .property-table {{
                font-size: 0.85em;
            }}
            
            .property-table th,
            .property-table td {{
                padding: 8px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Toronto Condo Investment Analysis</h1>
            <p class="subtitle">{datetime.now().strftime('%B %d, %Y')}</p>
            <div class="params">
                <div class="param">Mortgage: {self.mortgage_rate * 100:.1f}%</div>
                <div class="param">Property Tax: {self.property_tax_rate * 100:.1f}%</div>
                <div class="param">Down Payment: {self.down_payment_percent * 100:.0f}%</div>
                <div class="param">Min Price: ${self.min_price:,.0f}</div>
            </div>
        </div>
        
        <div class="content">
            <!-- Summary Cards -->
            <div class="summary-cards">
                <div class="summary-card">
                    <h3>Total Properties</h3>
                    <div class="value">{stats['total_properties']:,}</div>
                    <div class="subvalue">Analyzed (≥${self.min_price:,.0f})</div>
                </div>
                
                <div class="summary-card highlight">
                    <h3>Good Investments</h3>
                    <div class="value">{stats['good_investments']:,}</div>
                    <div class="subvalue">{stats['good_percentage']:.1f}% of total</div>
                </div>
                
                <div class="summary-card">
                    <h3>Avg Price (All)</h3>
                    <div class="value">${stats['avg_price_all']/1000:.0f}K</div>
                    <div class="subvalue">Median: ${stats['median_price_all']/1000:.0f}K</div>
                </div>
                
                <div class="summary-card">
                    <h3>Avg Net Yield</h3>
                    <div class="value">{stats['avg_net_yield_all']:.2f}%</div>
                    <div class="subvalue">Good: {stats['avg_net_yield_good']:.2f}%</div>
                </div>
                
                <div class="summary-card">
                    <h3>Avg Gross Yield</h3>
                    <div class="value">{stats['avg_gross_yield_all']:.2f}%</div>
                    <div class="subvalue">Good: {stats['avg_gross_yield_good']:.2f}%</div>
                </div>
                
                <div class="summary-card">
                    <h3>Avg Cash Return</h3>
                    <div class="value">{stats['avg_cash_return_all']:.1f}%</div>
                    <div class="subvalue">Good: {stats['avg_cash_return_good']:.1f}%</div>
                </div>
            </div>
            
            <!-- Charts -->
            <div class="chart-section">
                <div class="chart-grid">
                    <div class="chart-container">
                        <h3>Gross Rental Yield Distribution</h3>
                        <div style="position: relative; height: 280px;">
                            <canvas id="grossYieldChart"></canvas>
                        </div>
                    </div>
                    
                    <div class="chart-container">
                        <h3>Net Rental Yield Distribution</h3>
                        <div style="position: relative; height: 280px;">
                            <canvas id="netYieldChart"></canvas>
                        </div>
                    </div>
                    
                    <div class="chart-container">
                        <h3>Cash-on-Cash Return</h3>
                        <div style="position: relative; height: 280px;">
                            <canvas id="cashReturnChart"></canvas>
                        </div>
                    </div>
                    
                    <div class="chart-container">
                        <h3>Top 10 Neighborhoods</h3>
                        <div style="position: relative; height: 280px;">
                            <canvas id="neighborhoodChart"></canvas>
                        </div>
                    </div>
                    
                    <div class="chart-container">
                        <h3>Price vs Net Yield</h3>
                        <div style="position: relative; height: 280px;">
                            <canvas id="scatterChart"></canvas>
                        </div>
                    </div>
                    
                    <div class="chart-container">
                        <h3>Investment Score Distribution</h3>
                        <div style="position: relative; height: 280px;">
                            <canvas id="scoreChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Top Properties Table -->
            <div class="table-section">
                <h2>Top 20 Investment Opportunities</h2>
                <p class="click-hint">Click on any row to expand calculation details</p>
                <table class="property-table" id="topPropertiesTable">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Address</th>
                            <th>Neighbourhood</th>
                            <th>Price</th>
                            <th>Beds</th>
                            <th>Gross Yield</th>
                            <th>Net Yield</th>
                            <th>Score</th>
                            <th>Link</th>
                        </tr>
                    </thead>
                    <tbody>
                        {top_properties_html}
                    </tbody>
                </table>
            </div>
            
            {f'''
            <div class="table-section">
                <h2>All Properties Meeting Investment Criteria</h2>
                <p class="click-hint">Properties with Gross Yield ≥ 7% and Net Yield ≥ 3% | Click rows to see calculations</p>
                <table class="property-table" id="investmentTable">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Address</th>
                            <th>Price</th>
                            <th>Beds</th>
                            <th>Gross Yield</th>
                            <th>Net Yield</th>
                            <th>Cash Return</th>
                            <th>Link</th>
                        </tr>
                    </thead>
                    <tbody>
                        {good_investments_html}
                    </tbody>
                </table>
            </div>
            ''' if len(good_investments) > 0 else ''}
        </div>
        
        <div class="footer">
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            <p style="margin-top: 10px; color: #555;">Toronto Condo Investment Analyzer v3.0</p>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #4caf50;"></div>
                    <span>Excellent (>4% Net Yield)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #2196f3;"></div>
                    <span>Good (3-4% Net Yield)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ff9800;"></div>
                    <span>Fair (<3% Net Yield)</span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Store properties data
        const propertiesData = {json.dumps(properties_data) if len(good_investments) > 0 else '[]'};
        const topPropertiesData = {json.dumps(top_properties_data)};
        
        // Function to format currency
        function formatCurrency(value) {{
            return new Intl.NumberFormat('en-US', {{
                style: 'currency',
                currency: 'USD',
                minimumFractionDigits: 0,
                maximumFractionDigits: 0
            }}).format(value);
        }}
        
        // Function to format percentage
        function formatPercent(value) {{
            return value.toFixed(2) + '%';
        }}
        
        // Toggle calculation breakdown
        function toggleBreakdown(propId, event) {{
            if (event.target.tagName === 'A') {{
                return; // Don't toggle if clicking on a link
            }}
            
            const breakdownRow = document.getElementById(propId + '_breakdown');
            const contentDiv = document.getElementById(propId + '_content');
            const clickedRow = event.currentTarget;
            
            if (breakdownRow.style.display === 'none') {{
                // Find property data - check both arrays
                let propData = propertiesData.find(p => p.id === propId);
                if (!propData) {{
                    propData = topPropertiesData.find(p => p.id === propId);
                }}
                
                if (propData) {{
                    // Generate and populate content
                    contentDiv.innerHTML = generateBreakdownHTML(propData);
                    contentDiv.classList.add('show');
                }}
                breakdownRow.style.display = 'table-row';
                clickedRow.classList.add('expanded');
            }} else {{
                breakdownRow.style.display = 'none';
                contentDiv.classList.remove('show');
                clickedRow.classList.remove('expanded');
            }}
        }}
        
        // Generate breakdown HTML
        function generateBreakdownHTML(prop) {{
            const grossYieldMet = prop.grossYield >= 7;
            const netYieldMet = prop.netYield >= 3;
            
            // Additional property details if available
            const propertyInfo = prop.url && prop.url !== '#' ? 
                `<div class="calc-line">
                    <span class="calc-label">Property Link:</span>
                    <span class="calc-value"><a href="${{prop.url}}" target="_blank" style="color: #64b5f6;">View on Condos.ca</a></span>
                </div>` : '';
            
            const scoreInfo = prop.investmentScore ? 
                `<div class="calc-line">
                    <span class="calc-label">Investment Score:</span>
                    <span class="calc-value ${{prop.investmentScore >= 20 ? 'positive' : prop.investmentScore >= 15 ? 'warning' : ''}}">${{prop.investmentScore.toFixed(1)}}</span>
                </div>` : '';
            
            const propertyDetails = (prop.neighbourhood || prop.beds || prop.sqft) ?
                `<div class="calc-line">
                    <span class="calc-label">Details:</span>
                    <span class="calc-value">${{[prop.beds, prop.sqft, prop.neighbourhood].filter(x => x && x !== 'N/A').join(' • ')}}</span>
                </div>` : '';
            
            return `
                <div class="calc-grid">
                    <div class="calc-section">
                        <h4>Property Investment Overview</h4>
                        <div class="calc-line">
                            <span class="calc-label">Purchase Price:</span>
                            <span class="calc-value">${{formatCurrency(prop.price)}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Down Payment ({self.down_payment_percent * 100:.0f}%):</span>
                            <span class="calc-value">${{formatCurrency(prop.downPayment)}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Loan Amount:</span>
                            <span class="calc-value">${{formatCurrency(prop.price - prop.downPayment)}}</span>
                        </div>
                        ${{propertyDetails}}
                        ${{propertyInfo}}
                    </div>
                    
                    <div class="calc-section">
                        <h4>Rental Income</h4>
                        <div class="calc-line">
                            <span class="calc-label">Estimated Monthly Rent:</span>
                            <span class="calc-value positive">${{formatCurrency(prop.monthlyRent)}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Annual Rental Income:</span>
                            <span class="calc-value positive">${{formatCurrency(prop.annualRent)}}</span>
                        </div>
                        <div class="calc-line highlight">
                            <span class="calc-label">Gross Rental Yield:</span>
                            <span class="calc-value positive">${{formatPercent(prop.grossYield)}}
                                <span class="${{grossYieldMet ? 'criteria-met' : 'criteria-not-met'}}">${{grossYieldMet ? '≥7%' : '<7%'}} </span>
                            </span>
                        </div>
                    </div>
                    
                    <div class="calc-section">
                        <h4>Monthly Expenses</h4>
                        <div class="calc-line">
                            <span class="calc-label">Mortgage Interest ({self.mortgage_rate * 100:.1f}%):</span>
                            <span class="calc-value negative">${{formatCurrency(prop.monthlyMortgage)}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Property Tax ({self.property_tax_rate * 100:.1f}%):</span>
                            <span class="calc-value negative">${{formatCurrency(prop.monthlyTax)}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Maintenance Fee:</span>
                            <span class="calc-value negative">${{formatCurrency(prop.monthlyMaintenance)}}</span>
                        </div>
                        <div class="calc-line total">
                            <span class="calc-label">Total Monthly Expenses:</span>
                            <span class="calc-value negative">${{formatCurrency(prop.monthlyTotalExpense)}}</span>
                        </div>
                    </div>
                    
                    <div class="calc-section">
                        <h4>Net Returns</h4>
                        <div class="calc-line">
                            <span class="calc-label">Monthly Net Income:</span>
                            <span class="calc-value ${{prop.monthlyNetIncome >= 0 ? 'positive' : 'negative'}}">${{formatCurrency(prop.monthlyNetIncome)}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Annual Net Income:</span>
                            <span class="calc-value ${{prop.annualNetIncome >= 0 ? 'positive' : 'negative'}}">${{formatCurrency(prop.annualNetIncome)}}</span>
                        </div>
                        <div class="calc-line highlight">
                            <span class="calc-label">Net Rental Yield:</span>
                            <span class="calc-value positive">${{formatPercent(prop.netYield)}}
                                <span class="${{netYieldMet ? 'criteria-met' : 'criteria-not-met'}}">${{netYieldMet ? '≥3%' : '<3%'}} </span>
                            </span>
                        </div>
                    </div>
                    
                    <div class="calc-section">
                        <h4>Investment Metrics</h4>
                        <div class="calc-line">
                            <span class="calc-label">Cash-on-Cash Return:</span>
                            <span class="calc-value ${{prop.cashReturn >= 0 ? 'positive' : 'warning'}}">${{formatPercent(prop.cashReturn)}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Expense Ratio:</span>
                            <span class="calc-value ${{prop.expenseRatio <= 50 ? 'positive' : prop.expenseRatio <= 70 ? 'warning' : 'negative'}}">${{formatPercent(prop.expenseRatio)}}</span>
                        </div>
                        ${{scoreInfo}}
                    </div>
                    
                    <div class="calc-section">
                        <h4>Calculation Formulas</h4>
                        <div class="calc-line" style="flex-direction: column; align-items: flex-start;">
                            <span class="calc-label" style="margin-bottom: 5px;">Gross Yield = (Annual Rent ÷ Purchase Price) × 100</span>
                            <span class="calc-value" style="font-size: 0.85em;">= (${{prop.annualRent.toLocaleString()}} ÷ ${{prop.price.toLocaleString()}}) × 100 = ${{formatPercent(prop.grossYield)}}</span>
                        </div>
                        <div class="calc-line" style="flex-direction: column; align-items: flex-start; margin-top: 10px;">
                            <span class="calc-label" style="margin-bottom: 5px;">Net Yield = (Net Income ÷ Purchase Price) × 100</span>
                            <span class="calc-value" style="font-size: 0.85em;">= (${{prop.annualNetIncome.toLocaleString()}} ÷ ${{prop.price.toLocaleString()}}) × 100 = ${{formatPercent(prop.netYield)}}</span>
                        </div>
                        <div class="calc-line" style="flex-direction: column; align-items: flex-start; margin-top: 10px;">
                            <span class="calc-label" style="margin-bottom: 5px;">Cash-on-Cash = (Net Income ÷ Down Payment) × 100</span>
                            <span class="calc-value" style="font-size: 0.85em;">= (${{prop.annualNetIncome.toLocaleString()}} ÷ ${{prop.downPayment.toLocaleString()}}) × 100 = ${{formatPercent(prop.cashReturn)}}</span>
                        </div>
                    </div>
                </div>
            `;
        }}
        
        // Chart.js default settings for dark theme
        Chart.defaults.color = '#888';
        Chart.defaults.borderColor = '#333';
        
        const chartOptions = {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{
                    display: true,
                    position: 'top',
                    labels: {{
                        color: '#888'
                    }}
                }}
            }},
            scales: {{
                x: {{
                    ticks: {{ color: '#888' }},
                    grid: {{ color: '#2a2a2a' }}
                }},
                y: {{
                    ticks: {{ color: '#888' }},
                    grid: {{ color: '#2a2a2a' }}
                }}
            }}
        }};
        
        // Gross Yield Chart
        new Chart(document.getElementById('grossYieldChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(gross_yield_data['labels'])},
                datasets: [{{
                    label: 'Properties',
                    data: {json.dumps(gross_yield_data['values'])},
                    backgroundColor: 'rgba(33, 150, 243, 0.6)',
                    borderColor: 'rgba(33, 150, 243, 1)',
                    borderWidth: 1
                }}]
            }},
            options: chartOptions
        }});
        
        // Net Yield Chart
        new Chart(document.getElementById('netYieldChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(net_yield_data['labels'])},
                datasets: [{{
                    label: 'Properties',
                    data: {json.dumps(net_yield_data['values'])},
                    backgroundColor: 'rgba(76, 175, 80, 0.6)',
                    borderColor: 'rgba(76, 175, 80, 1)',
                    borderWidth: 1
                }}]
            }},
            options: chartOptions
        }});
        
        // Cash Return Chart
        new Chart(document.getElementById('cashReturnChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(cash_return_data['labels'])},
                datasets: [{{
                    label: 'Properties',
                    data: {json.dumps(cash_return_data['values'])},
                    backgroundColor: 'rgba(255, 152, 0, 0.6)',
                    borderColor: 'rgba(255, 152, 0, 1)',
                    borderWidth: 1
                }}]
            }},
            options: chartOptions
        }});
        
        // Neighborhood Chart
        new Chart(document.getElementById('neighborhoodChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(neighborhood_labels)},
                datasets: [{{
                    label: 'Listings',
                    data: {json.dumps(neighborhood_values)},
                    backgroundColor: 'rgba(156, 39, 176, 0.6)',
                    borderColor: 'rgba(156, 39, 176, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {{
                    legend: {{
                        display: true,
                        position: 'top',
                        labels: {{
                            color: '#888'
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        beginAtZero: true,
                        ticks: {{ 
                            color: '#888',
                            precision: 0,
                            stepSize: 1
                        }},
                        grid: {{ color: '#2a2a2a' }}
                    }},
                    y: {{
                        ticks: {{ color: '#888' }},
                        grid: {{ color: '#2a2a2a' }}
                    }}
                }}
            }}
        }});
        
        // Scatter Chart
        new Chart(document.getElementById('scatterChart'), {{
            type: 'scatter',
            data: {{
                datasets: [{{
                    label: 'Properties',
                    data: {json.dumps(scatter_points)},
                    backgroundColor: 'rgba(100, 181, 246, 0.5)',
                    borderColor: 'rgba(100, 181, 246, 1)',
                    borderWidth: 1,
                    pointRadius: 4
                }}]
            }},
            options: {{
                ...chartOptions,
                scales: {{
                    x: {{
                        type: 'linear',
                        position: 'bottom',
                        title: {{
                            display: true,
                            text: 'Price ($1000s)',
                            color: '#888'
                        }},
                        ticks: {{ color: '#888' }},
                        grid: {{ color: '#2a2a2a' }}
                    }},
                    y: {{
                        type: 'linear',
                        title: {{
                            display: true,
                            text: 'Net Rental Yield (%)',
                            color: '#888'
                        }},
                        ticks: {{ color: '#888' }},
                        grid: {{ color: '#2a2a2a' }}
                    }}
                }}
            }}
        }});
        
        // Investment Score Chart
        new Chart(document.getElementById('scoreChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(score_data['labels'])},
                datasets: [{{
                    label: 'Properties',
                    data: {json.dumps(score_data['values'])},
                    backgroundColor: 'rgba(255, 87, 34, 0.6)',
                    borderColor: 'rgba(255, 87, 34, 1)',
                    borderWidth: 1
                }}]
            }},
            options: chartOptions
        }});
    </script>
</body>
</html>
"""
        
        return html


# ============================================================================
# PART 3: MAIN PIPELINE FUNCTION
# ============================================================================

def run_pipeline():
    """
    Main pipeline function that runs scraping and analysis in sequence
    """
    print("\n" + "=" * 80)
    print("🏢 TORONTO CONDO INVESTMENT PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========== CONFIGURATION ==========
    # Scraping configuration
    BASE_URL = "https://condos.ca/toronto?sublocality_id=14&mode=Sale&page=2"
    START_PAGE = 1
    MAX_PAGES = 52  # Adjust this to control how many pages to scrape
    MAX_SEARCH_WORKERS = 20
    MAX_DETAIL_WORKERS = 64
    
    # Analysis configuration
    MORTGAGE_RATE = 3.5
    PROPERTY_TAX_RATE = 0.5
    DOWN_PAYMENT_PERCENT = 30
    MIN_GROSS_YIELD = 7
    MIN_NET_YIELD = 3
    MIN_PRICE = 200000
    
    # ========== STEP 1: WEB SCRAPING ==========
    print("\n" + "─" * 60)
    print("📥 STEP 1: SCRAPING CONDO DATA")
    print("─" * 60)
    
    condos_df = scrape_condos_fully_parallel(
        BASE_URL, 
        START_PAGE, 
        MAX_PAGES, 
        MAX_SEARCH_WORKERS,
        MAX_DETAIL_WORKERS
    )
    
    if condos_df.empty:
        print("\n❌ No data was collected. Exiting pipeline.")
        return None, None
    
    # Save raw scraped data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'toronto_condos_{timestamp}.csv'
    condos_df.to_csv(csv_filename, index=False, encoding='utf-8')
    print(f"\n💾 Raw data saved to: {csv_filename}")
    
    # ========== STEP 2: INVESTMENT ANALYSIS ==========
    print("\n" + "─" * 60)
    print("📊 STEP 2: INVESTMENT ANALYSIS")
    print("─" * 60)
    
    # Initialize analyzer with the scraped data
    analyzer = CondoInvestmentAnalyzer(
        condos_df,
        MORTGAGE_RATE, 
        PROPERTY_TAX_RATE,
        DOWN_PAYMENT_PERCENT,
        MIN_PRICE
    )
    
    # Calculate investment metrics
    analyzer.calculate_investment_metrics()
    
    # Filter for good investments
    good_investments = analyzer.filter_good_investments(MIN_GROSS_YIELD, MIN_NET_YIELD)
    
    # Generate HTML report
    report_filename = analyzer.generate_html_report(good_investments)
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Output Files:")
    print(f"   1. Raw Data CSV: {csv_filename}")
    print(f"   2. Investment Report: {report_filename}")
    print(f"\n🌐 Next Steps:")
    print(f"   Open {report_filename} in your browser to view the interactive report")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return csv_filename, report_filename


if __name__ == "__main__":
    # Check for required packages
    required_packages = ['pandas', 'numpy', 'bs4', 'cloudscraper']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️ Missing required packages: {', '.join(missing_packages)}")
        print(f"Please install them using: pip install {' '.join(missing_packages)}")
        sys.exit(1)
    
    # Run the pipeline
    try:
        csv_file, report_file = run_pipeline()
        
        # Optionally open the report automatically
        if report_file and os.path.exists(report_file):
            import webbrowser
            if input("\n📖 Open report in browser? (y/n): ").lower() == 'y':
                webbrowser.open(f'file://{os.path.abspath(report_file)}')
                
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()