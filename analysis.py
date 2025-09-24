import pandas as pd
import numpy as np
from datetime import datetime
import json
import re
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')
from scipy import stats

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

@dataclass
class MarketConfig:
    """Market configuration parameters for 2025 Toronto"""
    mortgage_rate: float = 5.5
    property_tax_rate: float = 0.7
    down_payment_percent: float = 20
    min_price: float = 300000
    vacancy_rate: float = 5
    insurance_monthly: float = 100
    property_management_rate: float = 8
    closing_costs_percent: float = 2
    annual_appreciation: float = 4.5  # Toronto historical average
    rent_growth_rate: float = 3.5
    amortization_years: int = 25  # More typical in Canada
    sale_costs_percent: float = 4.0  # Disposition costs at exit
    expense_growth_rate: float = 2.0  # Annual expense growth
    discount_rate: float = 6.0  # For NPV/IRR evaluation

class DataParser:
    """Enhanced data parsing utilities"""
    
    @staticmethod
    def extract_number(text: str, pattern: str = r'(\d+(?:,\d+)*(?:\.\d+)?)') -> Optional[float]:
        """Extract numeric value from text"""
        if pd.isna(text) or text == 'N/A':
            return None
        text = str(text).replace('$', '').replace(',', '')
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except:
                return None
        return None
    
    @staticmethod
    def parse_bedroom_config(beds_str: str) -> Dict:
        """Parse bedroom configuration into structured format (robust to formats)."""
        if pd.isna(beds_str) or beds_str == 'N/A':
            return {'beds': None, 'den': False, 'numeric': 0}

        text = str(beds_str).strip().lower()

        # Studio detection
        if 'studio' in text or text in {"st", "std"}:
            return {'beds': 0, 'den': False, 'numeric': 0.5}

        # Common tokens following numbers
        # Accept: bed, beds, bd, bdr, bdrm, bdrms (optional)
        # Detect forms like "2+1", "2+1bd", "2 bd", "2bdrm" etc.
        m = re.search(r'^(\d+)\s*(?:\+\s*(\d+))?\s*(?:beds?|bd|bdrm?s?)?$', text)
        if m:
            beds = int(m.group(1))
            den_rooms = int(m.group(2)) if m.group(2) else 0
            has_den = den_rooms > 0
            numeric = beds + (0.5 if has_den else 0)
            return {'beds': beds, 'den': has_den, 'numeric': numeric, 'den_rooms': den_rooms}

        # Fallback: just extract first integer
        m2 = re.search(r'(\d+)', text)
        if m2:
            beds = int(m2.group(1))
            return {'beds': beds, 'den': False, 'numeric': float(beds)}

        return {'beds': None, 'den': False, 'numeric': 0}
    
    @staticmethod
    def parse_time_period(time_str: str) -> Optional[int]:
        """Parse time period to days"""
        if pd.isna(time_str) or time_str == 'N/A':
            return None
        
        time_str = str(time_str).lower()
        
        time_units = {
            'minute': 0,
            'hour': 0,
            'day': 1,
            'week': 7,
            'month': 30,
            'year': 365
        }
        
        for unit, multiplier in time_units.items():
            if unit in time_str:
                match = re.search(r'(\d+)', time_str)
                if match:
                    return int(match.group(1)) * multiplier
                elif unit in ['minute', 'hour']:
                    return 0
                else:
                    return multiplier
        
        # Try to extract just a number
        match = re.search(r'(\d+)', time_str)
        if match:
            return int(match.group(1))
        
        return None

class MarketIntelligence:
    """Advanced market intelligence extraction and analysis"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.market_data = {}
        
    def extract_price_histories(self) -> Dict:
        """Extract and analyze price histories"""
        histories = {}
        
        for idx, row in self.df.iterrows():
            history = {'sales': [], 'leases': [], 'metrics': {}}
            
            # Parse sale history
            if pd.notna(row.get('price_history_for_sale')) and row['price_history_for_sale'] != 'N/A':
                try:
                    sales = json.loads(row['price_history_for_sale'])
                    if sales:
                        history['sales'] = sales
                        # Calculate price trajectory
                        prices = [DataParser.extract_number(s.get('price', 0)) for s in sales]
                        prices = [p for p in prices if p]
                        if len(prices) > 1:
                            history['metrics']['price_trend'] = (prices[0] - prices[-1]) / prices[-1] * 100 if prices[-1] else 0
                            history['metrics']['price_volatility'] = np.std(prices) / np.mean(prices) * 100 if prices else 0
                except:
                    pass
            
            # Parse lease history
            if pd.notna(row.get('price_history_for_lease')) and row['price_history_for_lease'] != 'N/A':
                try:
                    leases = json.loads(row['price_history_for_lease'])
                    if leases:
                        history['leases'] = leases
                        # Calculate rental trends
                        rents = [DataParser.extract_number(l.get('price', 0)) for l in leases]
                        rents = [r for r in rents if r and 500 < r < 10000]  # Filter reasonable rents
                        if rents:
                            history['metrics']['avg_rent'] = np.mean(rents)
                            history['metrics']['rent_stability'] = 100 - (np.std(rents) / np.mean(rents) * 100) if rents else 0
                except:
                    pass
            
            if history['sales'] or history['leases']:
                histories[idx] = history
        
        return histories
    
    def analyze_comparable_properties(self) -> Dict:
        """Analyze similar and nearby listings for market positioning"""
        comparables = {}
        
        for idx, row in self.df.iterrows():
            comp_data = {
                'similar': [],
                'nearby_sale': [],
                'nearby_rent': [],
                'metrics': {}
            }
            
            # Parse similar listings
            if pd.notna(row.get('similar_listings')) and row['similar_listings'] != 'N/A':
                try:
                    similar = json.loads(row['similar_listings'])
                    if similar:
                        comp_data['similar'] = similar
                        # Extract price points
                        prices = [DataParser.extract_number(s.get('price', '')) for s in similar]
                        prices = [p for p in prices if p and p > 100000]
                        if prices:
                            comp_data['metrics']['price_percentile'] = stats.percentileofscore(prices, row['price_numeric'])
                            comp_data['metrics']['price_vs_median'] = (row['price_numeric'] - np.median(prices)) / np.median(prices) * 100
                except:
                    pass
            
            # Parse nearby listings
            if pd.notna(row.get('nearby_listings_for_sale')) and row['nearby_listings_for_sale'] != 'N/A':
                try:
                    nearby_sale = json.loads(row['nearby_listings_for_sale'])
                    if nearby_sale:
                        comp_data['nearby_sale'] = nearby_sale
                except:
                    pass
            
            if pd.notna(row.get('nearby_listings_for_rent')) and row['nearby_listings_for_rent'] != 'N/A':
                try:
                    nearby_rent = json.loads(row['nearby_listings_for_rent'])
                    if nearby_rent:
                        comp_data['nearby_rent'] = nearby_rent
                        # Extract rental market data
                        rents = [DataParser.extract_number(r.get('price', '')) for r in nearby_rent]
                        rents = [r for r in rents if r and 500 < r < 10000]
                        if rents:
                            comp_data['metrics']['market_rent'] = np.median(rents)
                            comp_data['metrics']['rent_range'] = (min(rents), max(rents))
                except:
                    pass
            
            if any([comp_data['similar'], comp_data['nearby_sale'], comp_data['nearby_rent']]):
                comparables[idx] = comp_data
        
        return comparables
    
    def calculate_building_metrics(self) -> pd.DataFrame:
        """Calculate building-level metrics"""
        building_metrics = []
        
        for building, group in self.df.groupby('building_name'):
            if building and building != 'N/A':
                metrics = {
                    'building': building,
                    'unit_count': len(group),
                    'avg_price': group['price_numeric'].mean(),
                    'median_price': group['price_numeric'].median(),
                    'price_range': group['price_numeric'].max() - group['price_numeric'].min(),
                    'avg_maintenance': group['maintenance_monthly'].mean(),
                    'avg_sqft': group['sqft_clean'].mean(),
                    'avg_days_on_market': group['days_on_market_clean'].mean(),
                    'liquidity_score': 100 - min(group['days_on_market_clean'].mean() / 30 * 50, 100) if group['days_on_market_clean'].notna().any() else 50
                }
                
                # Calculate price per sqft if available
                valid_sqft = group[group['sqft_clean'].notna()]
                if len(valid_sqft) > 0:
                    metrics['avg_price_per_sqft'] = (valid_sqft['price_numeric'] / valid_sqft['sqft_clean']).mean()
                
                building_metrics.append(metrics)
        
        return pd.DataFrame(building_metrics)

class InvestmentAnalyzer:
    """Simplified investment analysis with clear rental estimation"""
    
    def __init__(self, df: pd.DataFrame, config: MarketConfig):
        self.df = df
        self.config = config
        self.market_intel = MarketIntelligence(df)
        
    def calculate_rental_income_simplified(self) -> pd.Series:
        """Simplified and clear rental income estimation with 3-step logic"""
        print("🧮 Using simplified 3-step rental estimation logic...")
        print("   1️⃣ Use lease history if available")
        print("   2️⃣ Use nearby rentals rent/sqft if available") 
        print("   3️⃣ Use neighborhood average rent/sqft as fallback")
        
        rental_estimates = []
        
        # Pre-calculate neighborhood rent/sqft averages (overall and by bedroom)
        neighborhood_rent_per_sqft = self._calculate_neighborhood_rent_per_sqft()
        neighborhood_rent_by_bed = self._calculate_neighborhood_rent_per_sqft_by_bed()
        
        for idx, row in self.df.iterrows():
            estimated_rent = None
            method_used = ""
            
            # Step 1: Check for lease history (highest priority - leased price > listed price)
            if pd.notna(row.get('price_history_for_lease')) and row['price_history_for_lease'] != 'N/A':
                lease_rent = self._extract_lease_history_rent(row)
                if lease_rent and 500 <= lease_rent <= 8000:
                    estimated_rent = lease_rent
                    method_used = f"📚 Lease history: ${estimated_rent:.0f}"
            
            # Step 2: Use nearby rentals to calculate rent/sqft
            if estimated_rent is None:
                nearby_rent = self._calculate_rent_from_nearby_listings(row)
                if nearby_rent and 500 <= nearby_rent <= 8000:
                    estimated_rent = nearby_rent
                    method_used = f"🏘️ Nearby rent/sqft: ${estimated_rent:.0f}"
            
            # Step 3: Use neighborhood average rent/sqft
            if estimated_rent is None:
                neighborhood = str(row.get('neighbourhood', '')).lower().strip()
                # Prefer bedroom-specific neighborhood center
                beds = row.get('bedroom_numeric', 1)
                avg_rpsf = None
                if neighborhood in neighborhood_rent_by_bed and beds in neighborhood_rent_by_bed[neighborhood]:
                    avg_rpsf = neighborhood_rent_by_bed[neighborhood][beds]
                elif neighborhood in neighborhood_rent_per_sqft:
                    avg_rpsf = neighborhood_rent_per_sqft[neighborhood]

                if avg_rpsf is not None:
                    property_sqft = row.get('sqft_clean', np.nan)
                    if pd.isna(property_sqft) or property_sqft <= 0:
                        property_sqft = self._estimate_sqft_from_beds(beds)
                    estimated_rent = property_sqft * avg_rpsf
                    method_used = f"🗺️ Neighborhood avg: ${estimated_rent:.0f}"
            
            # Final fallback: Basic calculation
            if estimated_rent is None:
                estimated_rent = self._basic_rent_calculation(row)
                method_used = f"⚙️ Basic calc: ${estimated_rent:.0f}"
            
            # Validate final estimate
            final_rent = self._validate_rent_range(estimated_rent, row)
            if abs(final_rent - estimated_rent) > 50:
                method_used += f" → ${final_rent:.0f} (validated)"
            
            print(f"Property {idx + 1}: {method_used}")
            rental_estimates.append(final_rent)
        
        return pd.Series(rental_estimates, index=self.df.index)
    
    def _extract_lease_history_rent(self, row: pd.Series) -> Optional[float]:
        """Extract rent from lease history - prioritize latest leased price, then latest listed price"""
        try:
            lease_history = json.loads(row['price_history_for_lease'])
            if not lease_history:
                return None

            # Separate leased and listed prices
            leased_prices = []
            listed_prices = []

            for lease in lease_history:
                # Check for actual leased price first (highest priority)
                leased_price = DataParser.extract_number(lease.get('leased_price', ''))
                if leased_price and 500 <= leased_price <= 8000:
                    # Weight by recency
                    date_str = lease.get('date', '') or lease.get('listing_date', '')
                    weight = self._calculate_recency_weight(date_str)
                    leased_prices.append((leased_price, weight))
                    continue

                # If no leased price, check for listed price
                listed_price = DataParser.extract_number(lease.get('listed_price', ''))
                if listed_price and 500 <= listed_price <= 8000:
                    # Weight by recency
                    date_str = lease.get('date', '') or lease.get('listing_date', '')
                    weight = self._calculate_recency_weight(date_str)
                    listed_prices.append((listed_price, weight))

            # Prioritize leased prices over listed prices
            if leased_prices:
                # Use latest leased price (highest weight)
                latest_leased = max(leased_prices, key=lambda x: x[1])
                return latest_leased[0]
            elif listed_prices:
                # Use latest listed price if no leased prices
                latest_listed = max(listed_prices, key=lambda x: x[1])
                return latest_listed[0]

        except Exception as e:
            print(f"Warning: Error parsing lease history: {e}")
            pass

        return None
    
    def _calculate_rent_from_nearby_listings(self, row: pd.Series) -> Optional[float]:
        """Calculate rent using nearby listings rent per sqft"""
        try:
            nearby_rents = json.loads(row['nearby_listings_for_rent'])
            if not nearby_rents:
                return None
            
            # Extract rent/sqft from nearby listings
            rent_per_sqft_data = []
            
            for rental in nearby_rents:
                rent_price = DataParser.extract_number(rental.get('price', ''))
                if not rent_price or rent_price < 500:
                    continue
                
                # Try to get sqft from rental listing (usually N/A, so estimate based on beds)
                rental_beds = self._parse_bedroom_count(rental.get('beds', ''))
                
                # Estimate sqft based on bedroom count if not available
                estimated_sqft = self._estimate_sqft_from_beds(rental_beds)
                
                if estimated_sqft > 0:
                    rent_per_sqft = rent_price / estimated_sqft
                    # Only use reasonable rent/sqft ratios
                    if 2.0 <= rent_per_sqft <= 8.0:  # $2-8 per sqft is reasonable for Toronto
                        rent_per_sqft_data.append(rent_per_sqft)
            
            if rent_per_sqft_data:
                # Robust center (median with IQR trimming)
                avg_rent_per_sqft = self._robust_center(rent_per_sqft_data)
                if avg_rent_per_sqft is None:
                    return None
                
                # Apply to current property's sqft (fallback to bedroom-based estimate)
                property_sqft = row.get('sqft_clean', np.nan)
                if pd.isna(property_sqft) or property_sqft <= 0:
                    property_sqft = self._estimate_sqft_from_beds(row.get('bedroom_numeric', 1))
                estimated_rent = property_sqft * avg_rent_per_sqft
                
                return estimated_rent
            
        except Exception:
            pass
        
        return None
    
    def _calculate_neighborhood_rent_per_sqft(self) -> Dict[str, float]:
        """Calculate average rent per sqft for each neighborhood"""
        neighborhood_data = {}
        
        # Collect all rent/sqft data by neighborhood
        for idx, row in self.df.iterrows():
            neighborhood = str(row.get('neighbourhood', '')).lower().strip()
            if not neighborhood or neighborhood == 'n/a':
                continue
            
            # Get rent data from nearby listings
            rent_sqft_pairs = self._extract_rent_sqft_from_nearby(row)
            
            if rent_sqft_pairs:
                if neighborhood not in neighborhood_data:
                    neighborhood_data[neighborhood] = []
                neighborhood_data[neighborhood].extend(rent_sqft_pairs)
        
        # Calculate robust centers
        neighborhood_averages = {}
        for neighborhood, rent_sqft_list in neighborhood_data.items():
            if len(rent_sqft_list) >= 3:  # Need at least 3 data points for reliability
                center = self._robust_center(rent_sqft_list)
                if center is not None and 2.0 <= center <= 8.0:
                    neighborhood_averages[neighborhood] = center
        
        return neighborhood_averages

    def _calculate_neighborhood_rent_per_sqft_by_bed(self) -> Dict[str, Dict[float, float]]:
        """Calculate neighborhood rent/sqft centers split by bedroom count."""
        data: Dict[str, Dict[float, List[float]]] = {}

        for idx, row in self.df.iterrows():
            neighborhood = str(row.get('neighbourhood', '')).lower().strip()
            if not neighborhood or neighborhood == 'n/a':
                continue
            try:
                nearby_rents = json.loads(row.get('nearby_listings_for_rent', '')) if pd.notna(row.get('nearby_listings_for_rent')) else []
            except Exception:
                nearby_rents = []

            for rental in nearby_rents:
                rent_price = DataParser.extract_number(rental.get('price', ''))
                if not rent_price or rent_price < 500:
                    continue
                rental_beds = self._parse_bedroom_count(rental.get('beds', ''))
                est_sqft = self._estimate_sqft_from_beds(rental_beds)
                if est_sqft > 0:
                    rpsf = rent_price / est_sqft
                    if 2.0 <= rpsf <= 8.0:
                        data.setdefault(neighborhood, {}).setdefault(rental_beds, []).append(rpsf)

        # Aggregate
        result: Dict[str, Dict[float, float]] = {}
        for nbh, by_bed in data.items():
            for bed, vals in by_bed.items():
                if len(vals) >= 3:
                    center = self._robust_center(vals)
                    if center is not None and 2.0 <= center <= 8.0:
                        result.setdefault(nbh, {})[bed] = center

        return result
    
    def _extract_rent_sqft_from_nearby(self, row: pd.Series) -> List[float]:
        """Extract rent per sqft data from nearby listings"""
        rent_sqft_data = []
        
        try:
            nearby_rents = json.loads(row['nearby_listings_for_rent'])
            for rental in nearby_rents:
                rent_price = DataParser.extract_number(rental.get('price', ''))
                if not rent_price or rent_price < 500:
                    continue
                
                # Estimate sqft based on bedroom count
                rental_beds = self._parse_bedroom_count(rental.get('beds', ''))
                estimated_sqft = self._estimate_sqft_from_beds(rental_beds)
                
                if estimated_sqft > 0:
                    rent_per_sqft = rent_price / estimated_sqft
                    if 2.0 <= rent_per_sqft <= 8.0:  # Reasonable range
                        rent_sqft_data.append(rent_per_sqft)
        except Exception:
            pass
        
        return rent_sqft_data
    
    def _estimate_sqft_from_beds(self, beds: float) -> float:
        """Estimate square footage based on bedroom count"""
        # Toronto condo averages
        sqft_estimates = {
            0.5: 450,   # Studio
            1: 550,     # 1BR
            1.5: 700,   # 1+1BR
            2: 850,     # 2BR
            2.5: 1000,  # 2+1BR
            3: 1200,    # 3BR
            3.5: 1400,  # 3+1BR
        }
        return sqft_estimates.get(beds, 600)  # Default 600 sqft
    
    def _basic_rent_calculation(self, row: pd.Series) -> float:
        """Basic rent calculation as final fallback"""
        beds = row.get('bedroom_numeric', 1)
        
        # Conservative base rents for Toronto 2025
        base_rents = {
            0.5: 1600,  # Studio
            1: 2000,    # 1BR  
            1.5: 2400,  # 1+1BR
            2: 2800,    # 2BR
            2.5: 3200,  # 2+1BR
            3: 3600,    # 3BR
        }
        
        base_rent = base_rents.get(beds, 2000)
        
        # Adjust for neighborhood
        neighborhood = str(row.get('neighbourhood', '')).lower()
        if any(area in neighborhood for area in ['king west', 'yorkville', 'bay street']):
            base_rent *= 1.2  # Premium areas
        elif any(area in neighborhood for area in ['downtown', 'core']):
            base_rent *= 1.1  # Central areas
        
        return base_rent

    def _robust_center(self, values: List[float]) -> Optional[float]:
        """Compute a robust center (median with IQR outlier trimming)."""
        if not values:
            return None
        arr = np.array(values, dtype=float)
        if arr.size < 3:
            return float(np.median(arr))
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        trimmed = arr[(arr >= lo) & (arr <= hi)]
        if trimmed.size == 0:
            return float(np.median(arr))
        return float(np.median(trimmed))
    
    def _extract_actual_rental_data(self) -> List[Dict]:
        """Extract actual rental data from CSV for ML training with enhanced parsing"""
        rental_data = []

        for idx, row in self.df.iterrows():
            current_price = row['price_numeric']
            current_beds = row.get('bedroom_numeric', 1)
            current_baths = row.get('baths_clean', 1)
            current_sqft = row.get('sqft_clean', 600) if pd.notna(row.get('sqft_clean')) else 600
            current_neighborhood = row.get('neighbourhood', '')

            # Extract from nearby rental listings with detailed parsing
            if pd.notna(row.get('nearby_listings_for_rent')) and row['nearby_listings_for_rent'] != 'N/A':
                try:
                    nearby_rents = json.loads(row['nearby_listings_for_rent'])
                    for rental in nearby_rents:
                        rent_info = self._parse_rental_listing(rental, current_price, current_beds, current_baths, current_sqft, current_neighborhood)
                        if rent_info:
                            rental_data.append(rent_info)
                except Exception as e:
                    # Try alternative parsing if JSON fails
                    alt_rents = self._parse_alternative_rental_format(row['nearby_listings_for_rent'])
                    for rent_info in alt_rents:
                        rental_data.append(rent_info)

            # Extract from lease history with recency weighting
            if pd.notna(row.get('price_history_for_lease')) and row['price_history_for_lease'] != 'N/A':
                try:
                    lease_history = json.loads(row['price_history_for_lease'])
                    for lease in lease_history:
                        rent_info = self._parse_lease_history(lease, row)
                        if rent_info:
                            rental_data.append(rent_info)
                except Exception as e:
                    # Try alternative parsing
                    alt_leases = self._parse_alternative_lease_format(row['price_history_for_lease'])
                    for rent_info in alt_leases:
                        rental_data.append(rent_info)

        # Remove duplicates and filter unreasonable data
        filtered_data = []
        seen = set()
        for item in rental_data:
            key = (item['price'], item['beds_numeric'], item['rent'])
            if key not in seen and 500 <= item['rent'] <= 15000:
                filtered_data.append(item)
                seen.add(key)

        return filtered_data

    def _parse_rental_listing(self, rental: Dict, current_price: float, current_beds: float,
                             current_baths: int, current_sqft: float, current_neighborhood: str) -> Optional[Dict]:
        """Parse individual rental listing with detailed information"""
        try:
            # Extract rent price
            rent_price = DataParser.extract_number(rental.get('price', ''))
            if not rent_price or rent_price < 500:
                return None

            # Extract rental property details
            beds_str = rental.get('beds', '')
            rental_beds = self._parse_bedroom_count(beds_str)
            if rental_beds == 0:  # Skip if we can't determine bedroom count
                return None

            baths_str = rental.get('baths', '')
            rental_baths = DataParser.extract_number(baths_str) or 1

            parking_str = rental.get('parking', '')
            rental_parking = 1 if 'parking' in parking_str.lower() and '0' not in parking_str else 0

            # Calculate similarity score for this rental to current property
            bed_similarity = 1.0 if rental_beds == current_beds else 0.8 if abs(rental_beds - current_beds) <= 0.5 else 0.5
            bath_similarity = 1.0 if rental_baths == current_baths else 0.8

            return {
                'price': current_price,
                'beds_numeric': current_beds,
                'baths': current_baths,
                'parking': rental_parking,  # Use rental's parking info
                'sqft': current_sqft,
                'maintenance': 300,  # Default, will be refined
                'neighborhood_score': self._calculate_neighborhood_score(current_neighborhood),
                'building_age_score': 0.7,  # Default for rental comparables
                'rent': rent_price,
                'similarity_score': (bed_similarity + bath_similarity) / 2,
                'rental_beds': rental_beds,
                'rental_baths': rental_baths,
                'data_source': 'nearby_rental'
            }
        except:
            return None

    def _parse_lease_history(self, lease: Dict, row: pd.Series) -> Optional[Dict]:
        """Parse lease history entry with date weighting"""
        try:
            rent_price = DataParser.extract_number(lease.get('price', ''))
            if not rent_price or rent_price < 500:
                return None

            # Calculate recency weight (newer leases are more relevant)
            listing_date_str = lease.get('listing_date', '')
            recency_weight = self._calculate_recency_weight(listing_date_str)

            return {
                'price': row['price_numeric'],
                'beds_numeric': row.get('bedroom_numeric', 1),
                'baths': row.get('baths_clean', 1),
                'parking': row.get('parking_clean', 0),
                'sqft': row.get('sqft_clean', 600) if pd.notna(row.get('sqft_clean')) else 600,
                'maintenance': row.get('maintenance_monthly', 300),
                'neighborhood_score': self._calculate_neighborhood_score(row.get('neighbourhood', '')),
                'building_age_score': self._calculate_building_age_score(row.get('age_of_building', '')),
                'rent': rent_price,
                'recency_weight': recency_weight,
                'data_source': 'lease_history'
            }
        except:
            return None

    def _parse_bedroom_count(self, beds_str: str) -> float:
        """Parse bedroom count from rental listing string"""
        if not beds_str:
            return 0

        s = str(beds_str).lower().strip()

        # Handle common formats
        if 'studio' in s:
            return 0.5

        # e.g., "2+1", "1 + 1 bd"
        m_plus = re.search(r'^(\d+)\s*\+\s*(\d+)', s)
        if m_plus:
            base = float(m_plus.group(1))
            return base + 0.5

        # e.g., "2bd", "3 bdrm", "1 bed"
        m_full = re.search(r'(\d+)\s*(?:beds?|bd|bdrm?s?)', s)
        if m_full:
            return float(m_full.group(1))

        # Fallback: any number
        m_any = re.search(r'(\d+)', s)
        if m_any:
            return float(m_any.group(1))
        return 0

    def _calculate_recency_weight(self, date_str: str) -> float:
        """Calculate weight based on recency using exponential decay (half-life ~ 1 year)."""
        if not date_str:
            return 0.5

        ds = str(date_str).strip()
        # Try multiple date formats
        fmt_candidates = [
            '%Y-%m-%d', '%d-%b-%Y', '%b %d, %Y', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d'
        ]
        dt = None
        for fmt in fmt_candidates:
            try:
                dt = datetime.strptime(ds, fmt)
                break
            except Exception:
                continue
        # If parsing fails, fallback to year substring heuristic
        if dt is None:
            try:
                year_m = re.search(r'(20\d{2})', ds)
                if year_m:
                    year = int(year_m.group(1))
                    # Map recent years to weights
                    now_y = datetime.today().year
                    delta_y = max(0, now_y - year)
                    return max(0.3, 1.0 * (0.5 ** delta_y))
            except Exception:
                return 0.5
            return 0.5

        # Compute exponential decay weight by days difference
        days = max(0, (datetime.today() - dt).days)
        # Half-life = 365 days
        weight = 0.5 ** (days / 365.0)
        return float(min(1.0, max(0.3, weight)))

    def _parse_alternative_rental_format(self, rental_data: str) -> List[Dict]:
        """Fallback parser for rental data if JSON parsing fails"""
        rentals = []
        try:
            # Extract prices using regex
            price_pattern = r'\$([0-9,]+)'
            prices = re.findall(price_pattern, rental_data)

            for price in prices:
                clean_price = int(price.replace(',', ''))
                if 500 <= clean_price <= 15000:
                    rentals.append({
                        'price': 0,  # Will be filled by caller
                        'beds_numeric': 1,  # Default
                        'baths': 1,
                        'parking': 0,
                        'sqft': 600,
                        'maintenance': 300,
                        'neighborhood_score': 0.5,
                        'building_age_score': 0.5,
                        'rent': clean_price,
                        'data_source': 'alternative_parsing'
                    })
        except:
            pass

        return rentals

    def _parse_alternative_lease_format(self, lease_data: str) -> List[Dict]:
        """Fallback parser for lease history if JSON parsing fails"""
        leases = []
        try:
            # Similar to rental parsing but for lease history
            price_pattern = r'(\d{3,5})'  # Look for 3-5 digit numbers (likely rents)
            prices = re.findall(price_pattern, lease_data)

            for price in prices:
                clean_price = int(price)
                if 500 <= clean_price <= 15000:
                    leases.append({
                        'price': 0,  # Will be filled by caller
                        'beds_numeric': 1,  # Default
                        'baths': 1,
                        'parking': 0,
                        'sqft': 600,
                        'maintenance': 300,
                        'neighborhood_score': 0.5,
                        'building_age_score': 0.5,
                        'rent': clean_price,
                        'data_source': 'alternative_lease_parsing'
                    })
        except:
            pass

        return leases

    def _extract_property_features(self, row: pd.Series) -> List[float]:
        """Extract feature vector for a property"""
        return [
                    row['price_numeric'],
                    row.get('bedroom_numeric', 1),
                    row.get('baths_clean', 1),
                    row.get('parking_clean', 0),
                    row.get('sqft_clean', 600) if pd.notna(row.get('sqft_clean')) else 600,
            row.get('maintenance_monthly', 300),
            self._calculate_neighborhood_score(row.get('neighbourhood', '')),
            self._calculate_building_age_score(row.get('age_of_building', ''))
        ]

    def _calculate_neighborhood_score(self, neighborhood: str) -> float:
        """Calculate neighborhood desirability score (0-1)"""
        if pd.isna(neighborhood) or neighborhood == 'N/A':
            return 0.5  # Neutral score

        neighborhood = str(neighborhood).lower()

        # Premium neighborhoods
        premium = ['yorkville', 'king west', 'liberty village', 'harbourfront', 'bay street',
                  'financial district', 'entertainment district', 'fashion district']
        if any(area in neighborhood for area in premium):
            return 0.9

        # High-demand neighborhoods
        high_demand = ['distillery', 'queen west', 'leslieville', 'junction', 'parkdale',
                      'roncesvalles', 'high park', 'the annex', 'kensington market']
        if any(area in neighborhood for area in high_demand):
            return 0.75

        # Standard neighborhoods
        standard = ['downtown', 'midtown', 'uptown', 'the danforth', 'greektown']
        if any(area in neighborhood for area in standard):
            return 0.6

        # Suburban areas
        suburban = ['scarborough', 'north york', 'etobicoke', 'east york', 'york']
        if any(area in neighborhood for area in suburban):
            return 0.4

        return 0.5  # Default

    def _calculate_building_age_score(self, age_str: str) -> float:
        """Calculate building quality score based on age (0-1)"""
        if pd.isna(age_str) or age_str == 'N/A':
            return 0.5  # Neutral score

        try:
            age = DataParser.extract_number(age_str)
            if age is None:
                return 0.5

            # New buildings (0-5 years) get highest score
            if age <= 5:
                return 0.9
            # Modern buildings (6-15 years) get high score
            elif age <= 15:
                return 0.8
            # Well-maintained buildings (16-25 years)
            elif age <= 25:
                return 0.7
            # Older buildings (26-40 years)
            elif age <= 40:
                return 0.6
            # Historic/very old buildings
            else:
                return 0.4
        except:
            return 0.5

    def _generate_enhanced_synthetic_rental_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate enhanced synthetic rental data with neighborhood and building factors"""
        # Enhanced Toronto rental market data by neighborhood and building type
        synthetic_data = [
            # [price, beds, baths, parking, sqft, maintenance, neighborhood_score, building_age_score, rent]
            # Downtown Premium
            [650000, 0.5, 1, 0, 450, 280, 0.9, 0.9, 2400],   # Studio
            [750000, 1, 1, 0, 550, 320, 0.9, 0.9, 2800],     # 1BR
            [900000, 1.5, 1, 1, 700, 380, 0.9, 0.9, 3400],   # 1+1BR
            [1100000, 2, 2, 1, 900, 450, 0.9, 0.9, 4200],    # 2BR
            [1300000, 2.5, 2, 1, 1100, 520, 0.9, 0.9, 4800], # 2+1BR
            [1600000, 3, 2, 2, 1400, 600, 0.9, 0.9, 5800],   # 3BR

            # Midtown/High-demand
            [550000, 1, 1, 0, 500, 300, 0.75, 0.8, 2200],    # 1BR
            [700000, 1.5, 1, 1, 650, 350, 0.75, 0.8, 2600],   # 1+1BR
            [850000, 2, 2, 1, 850, 400, 0.75, 0.8, 3200],     # 2BR
            [1000000, 2.5, 2, 1, 1000, 480, 0.75, 0.8, 3800], # 2+1BR

            # Suburban
            [450000, 1, 1, 1, 550, 280, 0.4, 0.6, 1800],     # 1BR
            [550000, 1.5, 1, 1, 700, 320, 0.4, 0.6, 2200],    # 1+1BR
            [650000, 2, 2, 1, 900, 380, 0.4, 0.6, 2600],      # 2BR
            [750000, 2.5, 2, 1, 1100, 420, 0.4, 0.6, 3000],   # 2+1BR
        ]

        # Add variations with realistic noise
        X = []
        y = []
        for base_data in synthetic_data:
            for _ in range(3):  # 3 variations each
                variation = base_data.copy()
                # Add realistic noise
                variation[0] *= np.random.uniform(0.92, 1.08)  # Price ±8%
                variation[4] *= np.random.uniform(0.97, 1.03)  # Sqft ±3%
                variation[5] *= np.random.uniform(0.95, 1.05)  # Maintenance ±5%
                variation[8] *= np.random.uniform(0.96, 1.04)  # Rent ±4%

                X.append(variation[:-1])  # All features except rent
                y.append(variation[-1])   # Rent
        
        return np.array(X), np.array(y)
    
    def _estimate_rental_income_enhanced(self) -> pd.Series:
        """Enhanced rule-based rental income estimation with proper prioritization"""
        print("🧮 Using enhanced rule-based rental estimation with proper validation...")
        rental_income = []

        # Get all rental comparables for quick lookup
        all_rental_comparables = self._get_all_rental_comparables()
        
        for idx, row in self.df.iterrows():
            base_rent = None
            beds = row.get('bedroom_numeric', 1)

            # Priority 1: Historical lease data for this exact property (leased price > listed price)
            historical_rent = self._get_historical_lease_rent(row)
            if historical_rent and self._is_reasonable_rent(historical_rent, row):
                base_rent = historical_rent
                print(f"📚 Using historical lease data for property {idx + 1}: ${base_rent:.0f}/month")

            # Priority 2: Direct comparable from market data (with strict bedroom matching)
            if base_rent is None:
                direct_comparable_rent = self._find_direct_comparable_rent(row, all_rental_comparables)
                if direct_comparable_rent and self._is_reasonable_rent(direct_comparable_rent, row):
                    base_rent = direct_comparable_rent
                    print(f"🎯 Found direct comparable for property {idx + 1}: ${base_rent:.0f}/month")

            # Priority 3: Enhanced calculation as fallback
            if base_rent is None:
                base_rent = self._calculate_base_rent_enhanced(row)

                # Apply neighborhood adjustments
                neighborhood_multiplier = self._calculate_neighborhood_multiplier(row.get('neighbourhood', ''))
                base_rent *= neighborhood_multiplier

                # Apply building quality adjustments
                building_multiplier = self._calculate_building_multiplier(row)
                base_rent *= building_multiplier

                # Apply property-specific adjustments
                property_adjustment = self._calculate_property_adjustments(row)
                base_rent *= property_adjustment

                # Apply market-based adjustments using comparable data
                market_adjustment = self._calculate_market_adjustment(row, idx)
                base_rent *= market_adjustment

            # Final validation and reasonable bounds
            base_rent = self._validate_rent_range(base_rent, row)

            rental_income.append(base_rent)

        return pd.Series(rental_income, index=self.df.index)

    def _get_historical_lease_rent(self, row: pd.Series) -> Optional[float]:
        """Extract historical lease rent for the same property - prioritize latest leased price, then latest listed price"""
        if pd.notna(row.get('price_history_for_lease')) and row['price_history_for_lease'] != 'N/A':
            try:
                lease_history = json.loads(row['price_history_for_lease'])
                if lease_history:
                    # Separate leased and listed prices
                    leased_prices = []
                    listed_prices = []

                    for lease in lease_history:
                        # Check for actual leased price first (highest priority)
                        leased_price = DataParser.extract_number(lease.get('leased_price', ''))
                        if leased_price and 500 <= leased_price <= 8000:
                            # Weight by recency
                            date_str = lease.get('date', '') or lease.get('listing_date', '')
                            weight = self._calculate_recency_weight(date_str)
                            leased_prices.append((leased_price, weight))
                            continue

                        # If no leased price, check for listed price
                        listed_price = DataParser.extract_number(lease.get('listed_price', ''))
                        if listed_price and 500 <= listed_price <= 8000:
                            # Weight by recency
                            date_str = lease.get('date', '') or lease.get('listing_date', '')
                            weight = self._calculate_recency_weight(date_str)
                            listed_prices.append((listed_price, weight))

                    # Prioritize leased prices over listed prices
                    if leased_prices:
                        # Use latest leased price (highest weight)
                        latest_leased = max(leased_prices, key=lambda x: x[1])
                        return latest_leased[0]
                    elif listed_prices:
                        # Use latest listed price if no leased prices
                        latest_listed = max(listed_prices, key=lambda x: x[1])
                        return latest_listed[0]

            except Exception:
                # Try regex fallback for malformed JSON - prioritize leased_price
                lease_data = str(row['price_history_for_lease'])
                # First try to find leased prices
                leased_prices = re.findall(r'"leased_price":\s*"(\d+)"', lease_data)
                if leased_prices:
                    numeric_prices = [int(p) for p in leased_prices if 500 <= int(p) <= 8000]
                    if numeric_prices:
                        return np.median(numeric_prices)

                # Fallback to listed prices
                listed_prices = re.findall(r'"listed_price":\s*"(\d+)"', lease_data)
                if listed_prices:
                    numeric_prices = [int(p) for p in listed_prices if 500 <= int(p) <= 8000]
                    if numeric_prices:
                        return np.median(numeric_prices)
        return None

    def _is_reasonable_rent(self, rent: float, row: pd.Series) -> bool:
        """Check if a rent estimate is reasonable for the property"""
        beds = row.get('bedroom_numeric', 1)
        price = row['price_numeric']

        # Basic reasonableness checks
        if rent < 500 or rent > 15000:
            return False

        # Bedroom-specific maximums (based on Toronto market data)
        max_rents = {
            0.5: 2200,  # Studio: max $2,200
            1: 3200,    # 1BR: max $3,200
            1.5: 3800,  # 1+1BR: max $3,800
            2: 4500,    # 2BR: max $4,500
            2.5: 5000,  # 2+1BR: max $5,000
            3: 6000,    # 3BR: max $6,000
        }

        max_reasonable = max_rents.get(beds, 4000)
        if rent > max_reasonable:
            return False

        # Price-to-rent ratio check (should be reasonable)
        monthly_price_ratio = price / rent
        if monthly_price_ratio < 80 or monthly_price_ratio > 600:  # Very extreme ratios
            return False

        return True

    def _validate_rent_range(self, rent: float, row: pd.Series) -> float:
        """Apply final validation and reasonable bounds to rent estimates"""
        beds = row.get('bedroom_numeric', 1)

        # Bedroom-specific reasonable ranges (Toronto market 2025)
        rent_ranges = {
            0.5: (900, 2200),   # Studio: $900-$2,200
            1: (1400, 3200),    # 1BR: $1,400-$3,200
            1.5: (1600, 3800),  # 1+1BR: $1,600-$3,800
            2: (1800, 4500),    # 2BR: $1,800-$4,500
            2.5: (2000, 5000),  # 2+1BR: $2,000-$5,000
            3: (2200, 6000),    # 3BR: $2,200-$6,000
        }

        min_rent, max_rent = rent_ranges.get(beds, (1200, 4000))

        # Clamp to reasonable range
        validated_rent = max(min_rent, min(rent, max_rent))

        # If we had to clamp significantly, it indicates an issue
        if abs(validated_rent - rent) / max(rent, 1) > 0.3:  # More than 30% adjustment
            print(f"⚠️  Rent for {beds}BR property adjusted from ${rent:.0f} to ${validated_rent:.0f}")

        return validated_rent

    def _calculate_neighborhood_rent_per_sqft_simple(self) -> Dict[str, float]:
        """Calculate average rent per sqft for each neighborhood using simple logic"""
        neighborhood_data = {}
        
        # Collect all rent/sqft data by neighborhood
        for idx, row in self.df.iterrows():
            neighborhood = str(row.get('neighbourhood', '')).lower().strip()
            if not neighborhood or neighborhood == 'n/a':
                continue
            
            # Get rent data from nearby listings
            rent_sqft_pairs = self._extract_rent_sqft_simple(row)
            
            if rent_sqft_pairs:
                if neighborhood not in neighborhood_data:
                    neighborhood_data[neighborhood] = []
                neighborhood_data[neighborhood].extend(rent_sqft_pairs)
        
        # Calculate robust centers
        neighborhood_averages = {}
        for neighborhood, rent_sqft_list in neighborhood_data.items():
            if len(rent_sqft_list) >= 3:  # Need at least 3 data points for reliability
                center = self._robust_center(rent_sqft_list)
                if center is not None and 2.0 <= center <= 8.0:
                    neighborhood_averages[neighborhood] = center
        
        return neighborhood_averages
    
    def _extract_lease_history_rent_simple(self, row: pd.Series) -> Optional[float]:
        """Extract rent from lease history - prioritize latest leased price, then latest listed price (simple version)"""
        try:
            lease_history = json.loads(row['price_history_for_lease'])
            if not lease_history:
                return None

            # Separate leased and listed prices
            leased_prices = []
            listed_prices = []

            for lease in lease_history:
                # Check for actual leased price first (highest priority)
                leased_price = DataParser.extract_number(lease.get('leased_price', ''))
                if leased_price and 500 <= leased_price <= 8000:
                    # Simple recency weighting
                    date_str = lease.get('date', '') or lease.get('listing_date', '')
                    weight = 1.0
                    if '2025' in date_str or '2024' in date_str:
                        weight = 1.0  # Recent
                    elif '2023' in date_str or '2022' in date_str:
                        weight = 0.9  # Somewhat recent
                    else:
                        weight = 0.7  # Older

                    leased_prices.append((leased_price, weight))
                    continue

                # If no leased price, check for listed price
                listed_price = DataParser.extract_number(lease.get('listed_price', ''))
                if listed_price and 500 <= listed_price <= 8000:
                    # Simple recency weighting
                    date_str = lease.get('date', '') or lease.get('listing_date', '')
                    weight = 1.0
                    if '2025' in date_str or '2024' in date_str:
                        weight = 1.0  # Recent
                    elif '2023' in date_str or '2022' in date_str:
                        weight = 0.9  # Somewhat recent
                    else:
                        weight = 0.7  # Older

                    listed_prices.append((listed_price, weight))

            # Prioritize leased prices over listed prices
            if leased_prices:
                # Use latest leased price (highest weight)
                latest_leased = max(leased_prices, key=lambda x: x[1])
                return latest_leased[0]
            elif listed_prices:
                # Use latest listed price if no leased prices
                latest_listed = max(listed_prices, key=lambda x: x[1])
                return latest_listed[0]

        except Exception as e:
            print(f"Warning: Error parsing lease history: {e}")
            pass

        return None
    
    def _calculate_rent_from_nearby_simple(self, row: pd.Series) -> Optional[float]:
        """Calculate rent using nearby listings rent per sqft - simple version"""
        try:
            if pd.isna(row.get('nearby_listings_for_rent')) or row['nearby_listings_for_rent'] == 'N/A':
                return None
                
            nearby_rents = json.loads(row['nearby_listings_for_rent'])
            if not nearby_rents:
                return None
            
            # Extract rent/sqft from nearby listings
            rent_per_sqft_data = []
            
            for rental in nearby_rents:
                rent_price = DataParser.extract_number(rental.get('price', ''))
                if not rent_price or rent_price < 500:
                    continue
                
                # Get bedroom count to estimate sqft
                rental_beds = self._parse_bedroom_count_simple(rental.get('beds', ''))
                estimated_sqft = self._estimate_sqft_simple(rental_beds)
                
                if estimated_sqft > 0:
                    rent_per_sqft = rent_price / estimated_sqft
                    # Only use reasonable rent/sqft ratios
                    if 2.0 <= rent_per_sqft <= 8.0:  # $2-8 per sqft is reasonable for Toronto
                        rent_per_sqft_data.append(rent_per_sqft)
            
            if rent_per_sqft_data:
                # Use robust center here as well
                avg_rent_per_sqft = self._robust_center(rent_per_sqft_data)
                if avg_rent_per_sqft is None:
                    return None
                
                # Apply to current property's sqft with better fallback
                property_sqft = row.get('sqft_clean', np.nan)
                if pd.isna(property_sqft) or property_sqft <= 0:
                    property_sqft = self._estimate_sqft_simple(row.get('bedroom_numeric', 1))
                estimated_rent = property_sqft * avg_rent_per_sqft
                
                return estimated_rent
            
        except Exception:
            pass
        
        return None
    
    def _extract_rent_sqft_simple(self, row: pd.Series) -> list:
        """Extract rent per sqft data from nearby listings - simple version"""
        rent_sqft_data = []
        
        try:
            if pd.isna(row.get('nearby_listings_for_rent')) or row['nearby_listings_for_rent'] == 'N/A':
                return []
                
            nearby_rents = json.loads(row['nearby_listings_for_rent'])
            for rental in nearby_rents:
                rent_price = DataParser.extract_number(rental.get('price', ''))
                if not rent_price or rent_price < 500:
                    continue
                
                # Estimate sqft based on bedroom count
                rental_beds = self._parse_bedroom_count_simple(rental.get('beds', ''))
                estimated_sqft = self._estimate_sqft_simple(rental_beds)
                
                if estimated_sqft > 0:
                    rent_per_sqft = rent_price / estimated_sqft
                    if 2.0 <= rent_per_sqft <= 8.0:  # Reasonable range
                        rent_sqft_data.append(rent_per_sqft)
        except Exception:
            pass
        
        return rent_sqft_data
    
    def _parse_bedroom_count_simple(self, beds_str: str) -> float:
        """Parse bedroom count from rental listing string - simple version"""
        if not beds_str:
            return 1.0  # Default
        
        s = str(beds_str).lower().strip()
        
        # Handle common formats
        if 'studio' in s:
            return 0.5
        m_plus = re.search(r'^(\d+)\s*\+\s*(\d+)', s)
        if m_plus:
            base = float(m_plus.group(1))
            return base + 0.5
        m_full = re.search(r'(\d+)\s*(?:beds?|bd|bdrm?s?)', s)
        if m_full:
            return float(m_full.group(1))
        match = re.search(r'(\d+)', s)
        if match:
            return float(match.group(1))
        return 1.0  # Default
    
    def _estimate_sqft_simple(self, beds: float) -> float:
        """Estimate square footage based on bedroom count - simple version"""
        # Toronto condo averages
        sqft_estimates = {
            0.5: 450,   # Studio
            1: 550,     # 1BR
            1.5: 700,   # 1+1BR
            2: 850,     # 2BR
            2.5: 1000,  # 2+1BR
            3: 1200,    # 3BR
            3.5: 1400,  # 3+1BR
        }
        return sqft_estimates.get(beds, 600)  # Default 600 sqft
    
    def _basic_rent_calculation_simple(self, row: pd.Series) -> float:
        """Basic rent calculation as final fallback - simple version"""
        beds = row.get('bedroom_numeric', 1)
        
        # Conservative base rents for Toronto 2025
        base_rents = {
            0.5: 1600,  # Studio
            1: 2000,    # 1BR  
            1.5: 2400,  # 1+1BR
            2: 2800,    # 2BR
            2.5: 3200,  # 2+1BR
            3: 3600,    # 3BR
        }
        
        base_rent = base_rents.get(beds, 2000)
        
        # Simple neighborhood adjustment
        neighborhood = str(row.get('neighbourhood', '')).lower()
        if any(area in neighborhood for area in ['king west', 'yorkville', 'bay street']):
            base_rent *= 1.2  # Premium areas
        elif any(area in neighborhood for area in ['downtown', 'core']):
            base_rent *= 1.1  # Central areas
        
        return base_rent

    def _calculate_base_rent_enhanced(self, row: pd.Series) -> float:
        """Calculate base rent using price-to-rent ratios and property features"""
        price = row['price_numeric']

        # Toronto average price-to-rent ratios by property type (monthly rent as % of price)
        bed_config = row.get('bedroom_config', {})
        beds = bed_config.get('numeric', 1) if bed_config else 1
            
        # Base rent percentages (conservative estimates)
        rent_percentages = {
            0.5: 0.0042,  # Studio: 0.42% of price per month
            1: 0.0040,    # 1BR: 0.40%
            1.5: 0.0038,  # 1+1BR: 0.38%
            2: 0.0036,    # 2BR: 0.36%
            2.5: 0.0034,  # 2+1BR: 0.34%
            3: 0.0032,    # 3BR: 0.32%
            3.5: 0.0030,  # 3+1BR: 0.30%
        }

        base_percentage = rent_percentages.get(beds, 0.0040)
        base_rent = price * base_percentage

        # Adjust for square footage efficiency
        if pd.notna(row.get('sqft_clean')) and row['sqft_clean'] > 0:
            sqft = row['sqft_clean']
            # Optimal size adjustment (600-800 sqft is most efficient)
            if sqft < 600:
                size_factor = 0.85 + (sqft / 600) * 0.15  # Smaller units less efficient
            elif sqft <= 800:
                size_factor = 1.0  # Optimal range
            else:
                size_factor = 1.0 - ((sqft - 800) / 1000) * 0.1  # Larger units slightly less efficient
            base_rent *= size_factor

        return base_rent

    def _calculate_neighborhood_multiplier(self, neighborhood: str) -> float:
        """Calculate neighborhood-based rent multiplier"""
        if pd.isna(neighborhood) or neighborhood == 'N/A':
            return 1.0

        neighborhood = str(neighborhood).lower()

        # Premium downtown areas
        premium_areas = ['yorkville', 'king west', 'liberty village', 'harbourfront', 'bay street',
                        'financial district', 'entertainment district', 'fashion district']
        if any(area in neighborhood for area in premium_areas):
            return 1.25

        # High-demand urban neighborhoods
        high_demand = ['distillery', 'queen west', 'leslieville', 'junction', 'parkdale',
                      'roncesvalles', 'high park', 'the annex', 'kensington market']
        if any(area in neighborhood for area in high_demand):
            return 1.15

        # Standard urban areas
        standard = ['downtown', 'midtown', 'uptown', 'the danforth', 'greektown',
                   'cabbagetown', 'rosedale', 'summerhill']
        if any(area in neighborhood for area in standard):
            return 1.05

        # Suburban areas
        suburban = ['scarborough', 'north york', 'etobicoke', 'east york', 'york',
                   'markham', 'richmond hill', 'vaughan']
        if any(area in neighborhood for area in suburban):
            return 0.85

        return 1.0

    def _calculate_building_multiplier(self, row: pd.Series) -> float:
        """Calculate building quality multiplier based on age and amenities"""
        multiplier = 1.0

        # Age-based adjustment
        if pd.notna(row.get('age_of_building')) and row['age_of_building'] != 'N/A':
            try:
                age = DataParser.extract_number(row['age_of_building'])
                if age is not None:
                    if age <= 3:
                        multiplier *= 1.10  # New building premium
                    elif age <= 10:
                        multiplier *= 1.05  # Modern building
                    elif age <= 20:
                        multiplier *= 1.0   # Well-maintained
                    elif age <= 35:
                        multiplier *= 0.95  # Older but maintained
                    else:
                        multiplier *= 0.90  # Historic/older
            except:
                pass
            
        # Building name premium (luxury buildings)
        building_name = str(row.get('building_name', '')).lower()
        luxury_indicators = ['pinnacle', 'elite', 'plaza', 'tower', 'signature', 'grand',
                           'marina', 'harbour', 'yonge', 'bay', 'sheraton', 'hilton']
        if any(indicator in building_name for indicator in luxury_indicators):
            multiplier *= 1.08

        return multiplier

    def _calculate_property_adjustments(self, row: pd.Series) -> float:
        """Calculate property-specific adjustments"""
        adjustment = 1.0

        # Bathroom adjustment (more bathrooms = higher rent)
        baths = row.get('baths_clean', 1)
        if baths > 1:
            adjustment *= (1.0 + (baths - 1) * 0.05)  # 5% per extra bathroom

        # Parking premium
        parking = row.get('parking_clean', 0)
        if parking > 0:
            adjustment *= (1.0 + parking * 0.03)  # 3% per parking spot

        # Den adjustment (additional room)
        bed_config = row.get('bedroom_config', {})
        if bed_config and bed_config.get('den', False):
            adjustment *= 1.03  # 3% premium for den

        # Maintenance fee consideration (higher fees might indicate better amenities)
        maintenance = row.get('maintenance_monthly', 300)
        if maintenance > 500:
            adjustment *= 1.02  # Slight premium for high-maintenance buildings

        return adjustment

    def _calculate_market_adjustment(self, row: pd.Series, idx: int) -> float:
        """Calculate market-based adjustment using comparable data"""
        adjustment = 1.0

        # Use nearby rental data if available
        if pd.notna(row.get('nearby_listings_for_rent')) and row['nearby_listings_for_rent'] != 'N/A':
            try:
                nearby_rents = json.loads(row['nearby_listings_for_rent'])
                if nearby_rents:
                    # Calculate average rent of nearby similar properties
                    similar_rents = []
                    for rental in nearby_rents:
                        rent_price = DataParser.extract_number(rental.get('price', ''))
                        if rent_price and 800 <= rent_price <= 8000:
                            similar_rents.append(rent_price)

                    if similar_rents:
                        avg_nearby_rent = np.mean(similar_rents)
                        # Calculate expected rent based on property characteristics
                        beds = row.get('bedroom_numeric', 1)
                        baths = row.get('baths_clean', 1)
                        sqft = row.get('sqft_clean', 600)

                        # Simple rent estimation for comparison
                        expected_rent = (beds * 1200) + (baths * 300) + (sqft * 2.5)

                        # Adjust based on market comparison
                        market_ratio = avg_nearby_rent / expected_rent
                        adjustment = min(max(market_ratio, 0.8), 1.2)  # Limit adjustment to ±20%
            except:
                pass

        return adjustment

    def _get_all_rental_comparables(self) -> Dict[str, List[Dict]]:
        """Extract all rental comparables from the dataset for quick lookup"""
        comparables = {
            'by_neighborhood': {},
            'by_bedrooms': {},
            'by_price_range': {}
        }

        for idx, row in self.df.iterrows():
            neighborhood = str(row.get('neighbourhood', '')).lower().strip()
            beds = row.get('bedroom_numeric', 1)
            price = row['price_numeric']

            # Extract rental data from this property's nearby listings
            if pd.notna(row.get('nearby_listings_for_rent')) and row['nearby_listings_for_rent'] != 'N/A':
                try:
                    nearby_rents = json.loads(row['nearby_listings_for_rent'])
                    for rental in nearby_rents:
                        rent_price = DataParser.extract_number(rental.get('price', ''))
                        if rent_price and 500 <= rent_price <= 15000:
                            # Categorize by neighborhood
                            if neighborhood not in comparables['by_neighborhood']:
                                comparables['by_neighborhood'][neighborhood] = []
                            comparables['by_neighborhood'][neighborhood].append({
                                'rent': rent_price,
                                'beds': self._parse_bedroom_count(rental.get('beds', '')),
                                'current_property_beds': beds,
                                'price': price,
                                'similarity': 1.0 if beds == self._parse_bedroom_count(rental.get('beds', '')) else 0.8
                            })

                            # Categorize by bedroom count
                            bed_key = f"{beds}bed"
                            if bed_key not in comparables['by_bedrooms']:
                                comparables['by_bedrooms'][bed_key] = []
                            comparables['by_bedrooms'][bed_key].append(rent_price)

                            # Categorize by price range
                            price_range = self._get_price_range(price)
                            if price_range not in comparables['by_price_range']:
                                comparables['by_price_range'][price_range] = []
                            comparables['by_price_range'][price_range].append(rent_price)

                except:
                    pass

            # Extract from lease history
            if pd.notna(row.get('price_history_for_lease')) and row['price_history_for_lease'] != 'N/A':
                try:
                    lease_history = json.loads(row['price_history_for_lease'])
                    for lease in lease_history:
                        rent_price = DataParser.extract_number(lease.get('price', ''))
                        if rent_price and 500 <= rent_price <= 15000:
                            # Add to bedroom category (assume same as current property)
                            bed_key = f"{beds}bed"
                            if bed_key not in comparables['by_bedrooms']:
                                comparables['by_bedrooms'][bed_key] = []
                            comparables['by_bedrooms'][bed_key].append(rent_price)

                            # Add to price range
                            price_range = self._get_price_range(price)
                            if price_range not in comparables['by_price_range']:
                                comparables['by_price_range'][price_range] = []
                            comparables['by_price_range'][price_range].append(rent_price)

                except:
                    pass

        return comparables

    def _find_direct_comparable_rent(self, row: pd.Series, comparables: Dict) -> Optional[float]:
        """Find direct rental comparable for a property with strict bedroom matching"""
        neighborhood = str(row.get('neighbourhood', '')).lower().strip()
        beds = row.get('bedroom_numeric', 1)
        price = row['price_numeric']

        # Method 1: Exact neighborhood and bedroom match (highest priority)
        if neighborhood in comparables['by_neighborhood']:
            neighborhood_matches = comparables['by_neighborhood'][neighborhood]
            exact_matches = [item for item in neighborhood_matches if abs(item['beds'] - beds) <= 0.1]  # Very strict matching

            if exact_matches:
                rents = [item['rent'] for item in exact_matches]
                return np.median(rents)  # Use median for robustness

        # Method 2: Bedroom count match across all neighborhoods (only exact bedroom count)
        bed_key = f"{beds}bed"
        if bed_key in comparables['by_bedrooms'] and comparables['by_bedrooms'][bed_key]:
            bedroom_rents = comparables['by_bedrooms'][bed_key]
            return np.median(bedroom_rents)

        # Method 3: Check for studio-specific matching (studios should only match with studios)
        if beds == 0.5:  # Studio
            studio_rents = []
            for neighborhood_name, neighborhood_data in comparables['by_neighborhood'].items():
                studio_matches = [item for item in neighborhood_data if item['beds'] == 0.5]
                studio_rents.extend([item['rent'] for item in studio_matches])
            
            if studio_rents:
                return np.median(studio_rents)

        # Method 4: Price range match (only as last resort and with bedroom validation)
        price_range = self._get_price_range(price)
        if price_range in comparables['by_price_range'] and comparables['by_price_range'][price_range]:
            price_range_rents = comparables['by_price_range'][price_range]
            # For studios, apply a significant discount to prevent unrealistic rents
            base_rent = np.median(price_range_rents)
            if beds == 0.5:  # Studio
                return base_rent * 0.6  # Studios typically rent at 60% of average market rent
            elif beds == 1:
                return base_rent * 0.8  # 1BR at 80% of average
            else:
                return base_rent

        return None

    def _get_price_range(self, price: float) -> str:
        """Categorize price into ranges for comparison"""
        if price < 400000:
            return "under_400k"
        elif price < 600000:
            return "400k_600k"
        elif price < 800000:
            return "600k_800k"
        elif price < 1000000:
            return "800k_1M"
        elif price < 1500000:
            return "1M_1.5M"
        elif price < 2000000:
            return "1.5M_2M"
        else:
            return "over_2M"

    # Keep the old method for backward compatibility
    def _estimate_rental_income_rules(self) -> pd.Series:
        """Legacy rule-based rental income estimation"""
        return self._estimate_rental_income_enhanced()
    
    def calculate_advanced_metrics(self) -> pd.DataFrame:
        """Calculate comprehensive investment metrics"""
        metrics = self.df.copy()
        
        # Get rental income estimates using simplified clear logic
        print("\n💰 Calculating rental income using simplified 3-step logic...")
        
        # Step 1: Calculate neighborhood rent/sqft averages
        neighborhood_rent_per_sqft = self._calculate_neighborhood_rent_per_sqft_simple()
        
        # Step 2: Estimate rent for each property
        rental_estimates = []
        
        for idx, row in self.df.iterrows():
            estimated_rent = None
            method_used = ""
            
            # Priority 1: Use lease history if available (leased price > listed price)
            if pd.notna(row.get('price_history_for_lease')) and row['price_history_for_lease'] != 'N/A':
                lease_rent = self._extract_lease_history_rent_simple(row)
                if lease_rent and 500 <= lease_rent <= 8000:
                    estimated_rent = lease_rent
                    method_used = "📚 Lease history"
            
            # Priority 2: Use nearby rentals to calculate rent/sqft
            if estimated_rent is None:
                nearby_rent = self._calculate_rent_from_nearby_simple(row)
                if nearby_rent and 500 <= nearby_rent <= 8000:
                    estimated_rent = nearby_rent
                    method_used = "🏘️ Nearby rent/sqft"
            
            # Priority 3: Use neighborhood average rent/sqft
            if estimated_rent is None:
                neighborhood = str(row.get('neighbourhood', '')).lower().strip()
                if neighborhood in neighborhood_rent_per_sqft:
                    property_sqft = row.get('sqft_clean', np.nan)
                    if pd.isna(property_sqft) or property_sqft <= 0:
                        property_sqft = self._estimate_sqft_simple(row.get('bedroom_numeric', 1))
                    avg_rent_per_sqft = neighborhood_rent_per_sqft[neighborhood]
                    estimated_rent = property_sqft * avg_rent_per_sqft
                    method_used = "🗺️ Neighborhood avg"
            
            # Final fallback: Basic calculation
            if estimated_rent is None:
                estimated_rent = self._basic_rent_calculation_simple(row)
                method_used = "⚙️ Basic calc"
            
            # Validate final estimate
            final_rent = self._validate_rent_range(estimated_rent, row)
            
            rental_estimates.append(final_rent)
        
        metrics['monthly_rent'] = pd.Series(rental_estimates, index=self.df.index)
        print(f"✅ Completed rental estimation for {len(rental_estimates)} properties")
        metrics['effective_monthly_rent'] = metrics['monthly_rent'] * (1 - self.config.vacancy_rate / 100)
        metrics['annual_rental_income'] = metrics['effective_monthly_rent'] * 12
        
        # Calculate expenses
        metrics['down_payment'] = metrics['price_numeric'] * (self.config.down_payment_percent / 100)
        metrics['loan_amount'] = metrics['price_numeric'] - metrics['down_payment']
        
        # Monthly mortgage payment (P&I)
        r = self.config.mortgage_rate / 100 / 12  # Monthly rate
        n = self.config.amortization_years * 12
        # Handle r == 0 edge case
        metrics['monthly_mortgage'] = np.where(
            r > 0,
            metrics['loan_amount'] * (r * (1 + r)**n) / ((1 + r)**n - 1),
            metrics['loan_amount'] / n
        )
        
        # Other monthly expenses
        # Prefer actual annual taxes when available
        metrics['monthly_property_tax'] = np.where(
            metrics['annual_taxes'].notna() & (metrics['annual_taxes'] > 0),
            metrics['annual_taxes'] / 12.0,
            metrics['price_numeric'] * (self.config.property_tax_rate / 100 / 12)
        )
        metrics['monthly_insurance'] = self.config.insurance_monthly
        metrics['monthly_management'] = metrics['monthly_rent'] * (self.config.property_management_rate / 100)
        
        # Use actual maintenance fees
        metrics['monthly_maintenance'] = metrics['maintenance_monthly'].fillna(300)
        
        # Total expenses
        metrics['total_monthly_expenses'] = (
            metrics['monthly_mortgage'] +
            metrics['monthly_property_tax'] +
            metrics['monthly_insurance'] +
            metrics['monthly_management'] +
            metrics['monthly_maintenance']
        )
        
        # Cash flow
        metrics['monthly_cash_flow'] = metrics['effective_monthly_rent'] - metrics['total_monthly_expenses']
        metrics['annual_cash_flow'] = metrics['monthly_cash_flow'] * 12
        
        # NOI and investment returns
        metrics['noi_monthly'] = (
            metrics['effective_monthly_rent'] - (
                metrics['monthly_maintenance'] + metrics['monthly_property_tax'] + metrics['monthly_insurance'] + metrics['monthly_management']
            )
        )
        metrics['noi_annual'] = metrics['noi_monthly'] * 12
        metrics['cap_rate'] = metrics['noi_annual'] / metrics['price_numeric'] * 100

        # Include closing costs in initial investment for COC
        metrics['closing_costs_amount'] = metrics['price_numeric'] * (self.config.closing_costs_percent / 100)
        metrics['initial_investment'] = metrics['down_payment'] + metrics['closing_costs_amount']
        metrics['cash_on_cash_return'] = metrics['annual_cash_flow'] / metrics['initial_investment'] * 100
        
        metrics['gross_rent_multiplier'] = metrics['price_numeric'] / metrics['annual_rental_income']
        
        # Break-even analysis
        metrics['break_even_rent'] = metrics['total_monthly_expenses'] / (1 - self.config.vacancy_rate / 100)
        metrics['rent_coverage_ratio'] = metrics['monthly_rent'] / metrics['break_even_rent']
        
        # 1% rule check (monthly rent should be at least 1% of purchase price)
        metrics['one_percent_rule'] = metrics['monthly_rent'] / metrics['price_numeric'] * 100
        
        # Price to rent ratio (annual)
        metrics['price_to_rent_ratio'] = metrics['price_numeric'] / metrics['annual_rental_income']

        # DSCR and interest rate sensitivity (+100bp)
        metrics['annual_debt_service'] = metrics['monthly_mortgage'] * 12
        metrics['dscr'] = np.where(
            metrics['loan_amount'] > 0,
            metrics['noi_annual'] / metrics['annual_debt_service'],
            np.nan
        )

        r_up = (self.config.mortgage_rate + 1.0) / 100 / 12
        metrics['monthly_mortgage_up_100bp'] = np.where(
            r_up > 0,
            metrics['loan_amount'] * (r_up * (1 + r_up)**n) / ((1 + r_up)**n - 1),
            metrics['loan_amount'] / n
        )
        metrics['total_monthly_expenses_up_100bp'] = (
            metrics['monthly_mortgage_up_100bp'] +
            metrics['monthly_property_tax'] +
            metrics['monthly_insurance'] +
            metrics['monthly_management'] +
            metrics['monthly_maintenance']
        )
        metrics['monthly_cash_flow_up_100bp'] = metrics['effective_monthly_rent'] - metrics['total_monthly_expenses_up_100bp']
        metrics['dscr_up_100bp'] = np.where(
            metrics['loan_amount'] > 0,
            metrics['noi_annual'] / (metrics['monthly_mortgage_up_100bp'] * 12),
            np.nan
        )

        # Additional meaningful metrics
        # LTV
        metrics['ltv'] = np.where(metrics['price_numeric'] > 0, metrics['loan_amount'] / metrics['price_numeric'] * 100, np.nan)

        # Price and rent per sqft
        metrics['price_per_sqft'] = np.where(
            metrics['sqft_clean'].notna() & (metrics['sqft_clean'] > 0),
            metrics['price_numeric'] / metrics['sqft_clean'],
            np.nan
        )
        metrics['rent_per_sqft'] = np.where(
            metrics['sqft_clean'].notna() & (metrics['sqft_clean'] > 0),
            metrics['monthly_rent'] / metrics['sqft_clean'],
            np.nan
        )

        # Neighborhood PPSF z-score (relative pricing risk)
        try:
            ppsf_mean = metrics.groupby(metrics['neighbourhood'].str.lower())['price_per_sqft'].transform('mean')
            ppsf_std = metrics.groupby(metrics['neighbourhood'].str.lower())['price_per_sqft'].transform('std')
            metrics['ppsf_zscore_neighborhood'] = (metrics['price_per_sqft'] - ppsf_mean) / ppsf_std
        except Exception:
            metrics['ppsf_zscore_neighborhood'] = np.nan

        # Yield on cost (NOI / total cost)
        metrics['total_cost'] = metrics['price_numeric'] + metrics['closing_costs_amount']
        metrics['yield_on_cost'] = np.where(
            metrics['total_cost'] > 0,
            metrics['noi_annual'] / metrics['total_cost'] * 100,
            np.nan
        )

        # Break-even occupancy (fraction of gross rent needed to cover expenses)
        metrics['breakeven_occupancy'] = np.where(
            metrics['monthly_rent'] > 0,
            metrics['total_monthly_expenses'] / metrics['monthly_rent'],
            np.nan
        )

        # Principal paydown year 1 and ROE including principal
        r = self.config.mortgage_rate / 100 / 12
        n = self.config.amortization_years * 12
        def remaining_balance(loan):
            return self._remaining_balance(loan, r, n, 12)
        try:
            metrics['remaining_balance_year1'] = metrics['loan_amount'].apply(remaining_balance)
        except Exception:
            metrics['remaining_balance_year1'] = np.nan
        metrics['principal_paid_year1'] = metrics['loan_amount'] - metrics['remaining_balance_year1']
        metrics['roe_year1_total'] = np.where(
            metrics['initial_investment'] > 0,
            (metrics['annual_cash_flow'] + metrics['principal_paid_year1']) / metrics['initial_investment'] * 100,
            np.nan
        )

        # Max loan for DSCR >= 1.0 and down payment gap
        A = self._mortgage_factor(r, n)
        metrics['max_loan_for_dscr1'] = np.where(
            A > 0,
            np.maximum(metrics['noi_monthly'], 0) / A,
            np.nan
        )
        metrics['suggested_down_payment_dscr1'] = np.maximum(metrics['price_numeric'] - metrics['max_loan_for_dscr1'], 0)
        metrics['down_payment_gap_dscr1'] = metrics['suggested_down_payment_dscr1'] - metrics['down_payment']

        # Stress scenarios
        # Rent -10%
        eff_rent_minus10 = metrics['monthly_rent'] * (1 - self.config.vacancy_rate / 100) * 0.9
        noi_m_minus10 = eff_rent_minus10 - (metrics['monthly_maintenance'] + metrics['monthly_property_tax'] + metrics['monthly_insurance'] + metrics['monthly_management'] * 0.9)
        metrics['dscr_rent_minus10'] = np.where(metrics['annual_debt_service'] > 0, (noi_m_minus10 * 12) / metrics['annual_debt_service'], np.nan)
        metrics['monthly_cash_flow_rent_minus10'] = eff_rent_minus10 - metrics['total_monthly_expenses'] + (metrics['monthly_management'] - metrics['monthly_management'] * 0.9)

        # Vacancy 10%
        eff_rent_vac10 = metrics['monthly_rent'] * (1 - 0.10)
        noi_m_vac10 = eff_rent_vac10 - (metrics['monthly_maintenance'] + metrics['monthly_property_tax'] + metrics['monthly_insurance'] + metrics['monthly_management'])
        metrics['dscr_vacancy_10'] = np.where(metrics['annual_debt_service'] > 0, (noi_m_vac10 * 12) / metrics['annual_debt_service'], np.nan)
        metrics['monthly_cash_flow_vacancy_10'] = eff_rent_vac10 - metrics['total_monthly_expenses']

        # Maintenance +20%
        maintenance_up = metrics['monthly_maintenance'] * 1.2
        noi_m_maint20 = metrics['effective_monthly_rent'] - (maintenance_up + metrics['monthly_property_tax'] + metrics['monthly_insurance'] + metrics['monthly_management'])
        metrics['dscr_maint_plus20'] = np.where(metrics['annual_debt_service'] > 0, (noi_m_maint20 * 12) / metrics['annual_debt_service'], np.nan)
        metrics['monthly_cash_flow_maint_plus20'] = metrics['effective_monthly_rent'] - (metrics['total_monthly_expenses'] + (maintenance_up - metrics['monthly_maintenance']))

        # Five-year projection (pre-tax) IRR/NPV and equity
        irr_list = []
        npv_list = []
        equity5_list = []
        sale_price5_list = []
        rem_bal5_list = []
        net_sale5_list = []
        payback_years = []
        for _, row in metrics.iterrows():
            res = self._simulate_5y(row)
            irr_list.append(res.get('irr_5y'))
            npv_list.append(res.get('npv_5y'))
            equity5_list.append(res.get('projected_equity_5y'))
            sale_price5_list.append(res.get('sale_price_5y'))
            rem_bal5_list.append(res.get('remaining_balance_5y'))
            net_sale5_list.append(res.get('net_sale_proceeds_5y'))
            payback_years.append(res.get('payback_years'))
        metrics['irr_5y'] = irr_list
        metrics['npv_5y'] = npv_list
        metrics['projected_equity_5y'] = equity5_list
        metrics['sale_price_5y'] = sale_price5_list
        metrics['remaining_balance_5y'] = rem_bal5_list
        metrics['net_sale_proceeds_5y'] = net_sale5_list
        metrics['payback_years_cashflow_only'] = payback_years

        # Risk flags (string list)
        flags = []
        for _, row in metrics.iterrows():
            f = []
            if pd.notna(row.get('dscr')) and row['dscr'] < 1.0:
                f.append('DSCR<1')
            if pd.notna(row.get('rent_coverage_ratio')) and row['rent_coverage_ratio'] < 1.0:
                f.append('RentCoverage<1')
            if pd.notna(row.get('monthly_cash_flow_up_100bp')) and row['monthly_cash_flow_up_100bp'] < 0:
                f.append('RateShockCF<0')
            if pd.notna(row.get('ppsf_zscore_neighborhood')) and row['ppsf_zscore_neighborhood'] > 2.0:
                f.append('PPSF>+2σ')
            if pd.notna(row.get('ltv')) and row['ltv'] > 80:
                f.append('LTV>80%')
            flags.append(','.join(f))
        metrics['risk_flags'] = flags

        return metrics

    def _mortgage_factor(self, r: float, n: int) -> float:
        """Monthly payment per $1 loan (annuity factor)."""
        try:
            if r <= 0:
                return 1.0 / n if n > 0 else np.nan
            return r * (1 + r) ** n / ((1 + r) ** n - 1)
        except Exception:
            return np.nan

    def _remaining_balance(self, loan_amount: float, r: float, n: int, k: int) -> float:
        """Remaining balance after k payments for a fixed-rate mortgage."""
        try:
            if r <= 0:
                # Straight-line if zero rate
                return max(loan_amount - loan_amount * (k / n), 0)
            A = self._mortgage_factor(r, n)
            M = loan_amount * A
            # Remaining balance formula
            return loan_amount * (1 + r) ** k - M * ((1 + r) ** k - 1) / r
        except Exception:
            return np.nan

    def _npv(self, rate: float, cashflows: List[float]) -> float:
        try:
            return float(sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows)))
        except Exception:
            return np.nan

    def _irr(self, cashflows: List[float]) -> Optional[float]:
        """IRR via bisection on [-0.99, 1.5], returns None if no sign change."""
        try:
            # Ensure sign change
            f = lambda r: self._npv(r, cashflows)
            lo, hi = -0.99, 1.5
            flo, fhi = f(lo), f(hi)
            if np.isnan(flo) or np.isnan(fhi) or flo * fhi > 0:
                return None
            for _ in range(80):
                mid = (lo + hi) / 2
                fmid = f(mid)
                if abs(fmid) < 1e-6:
                    return float(mid * 100)
                if flo * fmid <= 0:
                    hi, fhi = mid, fmid
                else:
                    lo, flo = mid, fmid
            return float(((lo + hi) / 2) * 100)
        except Exception:
            return None

    def _simulate_5y(self, row: pd.Series) -> Dict[str, Any]:
        """Simple 5-year pro forma with appreciation, growth, sale costs; pre-tax."""
        try:
            price = float(row['price_numeric'])
            loan = float(row['loan_amount'])
            r_m = self.config.mortgage_rate / 100 / 12
            n = self.config.amortization_years * 12
            A = self._mortgage_factor(r_m, n)
            M = loan * A if not np.isnan(A) else 0.0

            rent_g = self.config.rent_growth_rate / 100
            exp_g = self.config.expense_growth_rate / 100
            disc = self.config.discount_rate / 100

            # Base monthly values at Year 1
            rent0 = float(row['monthly_rent'])
            vac = self.config.vacancy_rate / 100
            eff_rent0 = rent0 * (1 - vac)
            maint0 = float(row['monthly_maintenance'])
            tax0 = float(row['monthly_property_tax'])
            ins0 = float(row['monthly_insurance'])
            mgmt_rate = self.config.property_management_rate / 100
            mgmt0 = rent0 * mgmt_rate

            cashflows = [-float(row['initial_investment'])]
            cf_only = []
            balance = loan
            for year in range(1, 6):
                # escalate
                rent_y = rent0 * ((1 + rent_g) ** (year - 1))
                eff_y = rent_y * (1 - vac)
                maint_y = maint0 * ((1 + exp_g) ** (year - 1))
                tax_y = tax0 * ((1 + exp_g) ** (year - 1))
                ins_y = ins0 * ((1 + exp_g) ** (year - 1))
                mgmt_y = rent_y * mgmt_rate
                noi_m_y = eff_y - (maint_y + tax_y + ins_y + mgmt_y)
                cf_y = (noi_m_y - M) * 12

                if year < 5:
                    cashflows.append(cf_y)
                    cf_only.append(cf_y)
                else:
                    # Year 5 terminal value
                    sale_price_5 = price * ((1 + self.config.annual_appreciation / 100) ** 5)
                    rem_bal_5 = self._remaining_balance(loan, r_m, n, 12 * 5)
                    sale_costs_5 = sale_price_5 * (self.config.sale_costs_percent / 100)
                    net_sale = sale_price_5 - sale_costs_5 - rem_bal_5
                    cashflows.append(cf_y + net_sale)
                    cf_only.append(cf_y)
                    projected_equity_5 = net_sale

            npv_5 = self._npv(disc, cashflows)
            irr_5 = self._irr(cashflows)

            # Payback years (including terminal sale at year 5)
            cum = -float(row['initial_investment'])
            payback = None

            for y in range(1, 6):
                # add operating cash flows
                if y <= len(cf_only):
                    cum += cf_only[y-1]

                # At year 5, also add the terminal sale proceeds
                if y == 5:
                    sale_price_5 = price * ((1 + self.config.annual_appreciation / 100) ** 5)
                    rem_bal_5 = self._remaining_balance(loan, r_m, n, 12 * 5)
                    sale_costs_5 = sale_price_5 * (self.config.sale_costs_percent / 100)
                    net_sale = sale_price_5 - sale_costs_5 - rem_bal_5
                    cum += net_sale

                if cum >= 0 and payback is None:
                    payback = y

            return {
                'irr_5y': irr_5,
                'npv_5y': npv_5,
                'projected_equity_5y': projected_equity_5,
                'sale_price_5y': price * ((1 + self.config.annual_appreciation / 100) ** 5),
                'remaining_balance_5y': self._remaining_balance(loan, r_m, n, 60),
                'net_sale_proceeds_5y': (price * ((1 + self.config.annual_appreciation / 100) ** 5)) - (price * ((1 + self.config.annual_appreciation / 100) ** 5) * (self.config.sale_costs_percent / 100)) - self._remaining_balance(loan, r_m, n, 60),
                'payback_years': payback
            }
        except Exception:
            return {
                'irr_5y': None,
                'npv_5y': np.nan,
                'projected_equity_5y': np.nan,
                'sale_price_5y': np.nan,
                'remaining_balance_5y': np.nan,
                'net_sale_proceeds_5y': np.nan,
                'payback_years': None
            }
    
    def calculate_risk_scores(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive risk assessment"""
        risk_df = metrics_df.copy()
        
        # Initialize risk components
        risk_components = []
        
        # Market risk (based on days on market and price volatility)
        dom_risk = np.where(
            risk_df['days_on_market_clean'].notna(),
            risk_df['days_on_market_clean'] / 60 * 30,  # 60+ days = 30 risk points
            15  # Unknown = medium risk
        )
        risk_components.append(('market_risk', np.clip(dom_risk, 0, 30)))
        
        # Liquidity risk (ability to sell)
        liquidity_risk = np.where(
            risk_df['days_on_market_clean'].notna(),
            risk_df['days_on_market_clean'] / 90 * 25,  # 90+ days = 25 risk points
            12.5
        )
        risk_components.append(('liquidity_risk', np.clip(liquidity_risk, 0, 25)))
        
        # Cash flow risk
        cf_risk = np.where(
            risk_df['monthly_cash_flow'] < -500, 25,  # Significant negative
            np.where(risk_df['monthly_cash_flow'] < 0, 15,  # Negative
                    np.where(risk_df['monthly_cash_flow'] < 200, 10, 0))  # Low positive
        )
        risk_components.append(('cashflow_risk', cf_risk))
        
        # Concentration risk (building exposure)
        building_counts = risk_df['building_name'].value_counts()
        concentration_risk = risk_df['building_name'].map(
            lambda x: min(building_counts.get(x, 1) * 2, 20) if x != 'N/A' else 10
        )
        risk_components.append(('concentration_risk', concentration_risk))
        
        # Calculate total risk score
        total_risk = 0
        for name, component in risk_components:
            risk_df[name] = component
            total_risk += component
        
        risk_df['total_risk_score'] = total_risk
        
        # Risk categories
        risk_df['risk_category'] = pd.cut(
            risk_df['total_risk_score'],
            bins=[0, 30, 50, 70, 100],
            labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
        )
        
        return risk_df
    
    def identify_investment_strategies(self, metrics_df: pd.DataFrame) -> Dict:
        """Identify properties for different investment strategies"""
        strategies = {}
        
        # Buy and Hold (stable cash flow)
        strategies['buy_and_hold'] = metrics_df[
            (metrics_df['monthly_cash_flow'] > 200) &
            (metrics_df['cap_rate'] > 4) &
            (metrics_df['total_risk_score'] < 50)
        ].nlargest(10, 'annual_cash_flow')
        
        # Value Investing (underpriced)
        if 'price_percentile' in metrics_df.columns:
            strategies['value_investing'] = metrics_df[
                (metrics_df['price_percentile'] < 40) &
                (metrics_df['cap_rate'] > 3.5) &
                (metrics_df['sqft_clean'].notna())
            ].nlargest(10, 'cap_rate')
        
        # Growth Investing (appreciation potential)
        growth_neighborhoods = ['king west', 'liberty village', 'junction', 'leslieville']
        strategies['growth_investing'] = metrics_df[
            metrics_df['neighbourhood'].str.lower().apply(
                lambda x: any(area in str(x) for area in growth_neighborhoods)
            ) &
            (metrics_df['days_on_market_clean'] < 30)
        ].nlargest(10, 'price_numeric')
        
        # Cash Flow Maximization
        strategies['cash_flow_max'] = metrics_df[
            (metrics_df['monthly_cash_flow'] > 0)
        ].nlargest(10, 'monthly_cash_flow')
        
        # Luxury Investment
        strategies['luxury'] = metrics_df[
            (metrics_df['price_numeric'] > 800000) &
            (metrics_df['sqft_clean'] > 1000) &
            (metrics_df['parking_clean'] > 0)
        ].nlargest(10, 'price_numeric')
        
        # Starter Investment (low entry)
        strategies['starter'] = metrics_df[
            (metrics_df['price_numeric'] < 500000) &
            (metrics_df['monthly_cash_flow'] > -200) &
            (metrics_df['cap_rate'] > 3)
        ].nlargest(10, 'cap_rate')
        
        # Flip Candidates (quick sale potential)
        if 'price_vs_median' in metrics_df.columns:
            strategies['flip_candidates'] = metrics_df[
                (metrics_df['price_vs_median'] < -5) &  # 5% below median
                (metrics_df['days_on_market_clean'] < 14) &
                (metrics_df['total_risk_score'] < 50)
            ].nlargest(10, 'price_vs_median')
        
        return strategies

class ReportGenerator:
    """Advanced report generation with visualizations"""
    
    def __init__(self, analyzer: InvestmentAnalyzer, metrics_df: pd.DataFrame, config: MarketConfig = None):
        self.analyzer = analyzer
        self.metrics_df = metrics_df
        self.config = config or MarketConfig()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def generate_executive_summary(self) -> Dict:
        """Generate executive summary statistics"""
        summary = {
            'market_overview': {
                'total_properties': len(self.metrics_df),
                'avg_price': self.metrics_df['price_numeric'].mean(),
                'median_price': self.metrics_df['price_numeric'].median(),
                'price_range': (self.metrics_df['price_numeric'].min(), 
                               self.metrics_df['price_numeric'].max()),
                'avg_days_on_market': self.metrics_df['days_on_market_clean'].mean()
            },
            'investment_metrics': {
                'avg_cap_rate': self.metrics_df['cap_rate'].mean(),
                'avg_cash_on_cash': self.metrics_df['cash_on_cash_return'].mean(),
                'positive_cashflow_pct': (self.metrics_df['monthly_cash_flow'] > 0).mean() * 100,
                'avg_monthly_cashflow': self.metrics_df['monthly_cash_flow'].mean(),
                'best_cap_rate': self.metrics_df['cap_rate'].max(),
                'best_cashflow': self.metrics_df['monthly_cash_flow'].max()
            },
            'risk_profile': {
                'low_risk_pct': (self.metrics_df['risk_category'] == 'Low Risk').mean() * 100,
                'avg_risk_score': self.metrics_df['total_risk_score'].mean(),
                'safest_neighborhood': self._get_safest_neighborhood()
            }
        }
        return summary
    
    def _get_safest_neighborhood(self) -> str:
        """Identify neighborhood with lowest average risk"""
        neighborhood_risk = self.metrics_df.groupby('neighbourhood')['total_risk_score'].mean()
        return neighborhood_risk.idxmin() if not neighborhood_risk.empty else 'N/A'
    
    def _generate_property_calculation_data(self, row) -> Dict:
        """Generate calculation data for a property"""
        return {
            'price': row['price_numeric'],
            'down_payment': row['down_payment'],
            'loan_amount': row['loan_amount'],
            'monthly_rent': row['monthly_rent'],
            'effective_monthly_rent': row['effective_monthly_rent'],
            'annual_rental_income': row['annual_rental_income'],
            'monthly_mortgage': row['monthly_mortgage'],
            'monthly_property_tax': row['monthly_property_tax'],
            'monthly_insurance': row['monthly_insurance'],
            'monthly_management': row['monthly_management'],
            'monthly_maintenance': row['monthly_maintenance'],
            'total_monthly_expenses': row['total_monthly_expenses'],
            'monthly_cash_flow': row['monthly_cash_flow'],
            'annual_cash_flow': row['annual_cash_flow'],
            'cap_rate': row['cap_rate'],
            'cash_on_cash_return': row['cash_on_cash_return'],
            'gross_rent_multiplier': row['gross_rent_multiplier'],
            'break_even_rent': row['break_even_rent'],
            'rent_coverage_ratio': row['rent_coverage_ratio'],
            'one_percent_rule': row['one_percent_rule'],
            'risk_score': row['total_risk_score'],
            'risk_category': row['risk_category'],
            'down_payment_percent': (row['down_payment'] / row['price_numeric'] * 100) if row['price_numeric'] > 0 else 0,

            # Newly exposed metrics in UI
            'ltv': row.get('ltv', np.nan),
            'price_per_sqft': round(row.get('price_per_sqft', np.nan), 0) if pd.notna(row.get('price_per_sqft')) else np.nan,
            'rent_per_sqft': row.get('rent_per_sqft', np.nan),
            'ppsf_zscore_neighborhood': row.get('ppsf_zscore_neighborhood', np.nan),
            'yield_on_cost': row.get('yield_on_cost', np.nan),
            'breakeven_occupancy': row.get('breakeven_occupancy', np.nan),
            'principal_paid_year1': row.get('principal_paid_year1', np.nan),
            'roe_year1_total': row.get('roe_year1_total', np.nan),

            'dscr': row.get('dscr', np.nan),
            'dscr_up_100bp': row.get('dscr_up_100bp', np.nan),
            'monthly_cash_flow_up_100bp': row.get('monthly_cash_flow_up_100bp', np.nan),
            'max_loan_for_dscr1': row.get('max_loan_for_dscr1', np.nan),
            'suggested_down_payment_dscr1': row.get('suggested_down_payment_dscr1', np.nan),
            'down_payment_gap_dscr1': row.get('down_payment_gap_dscr1', np.nan),

            'dscr_rent_minus10': row.get('dscr_rent_minus10', np.nan),
            'monthly_cash_flow_rent_minus10': row.get('monthly_cash_flow_rent_minus10', np.nan),
            'dscr_vacancy_10': row.get('dscr_vacancy_10', np.nan),
            'monthly_cash_flow_vacancy_10': row.get('monthly_cash_flow_vacancy_10', np.nan),
            'dscr_maint_plus20': row.get('dscr_maint_plus20', np.nan),
            'monthly_cash_flow_maint_plus20': row.get('monthly_cash_flow_maint_plus20', np.nan),

            'irr_5y': row.get('irr_5y', None),
            'npv_5y': row.get('npv_5y', np.nan),
            'projected_equity_5y': row.get('projected_equity_5y', np.nan),
            'sale_price_5y': row.get('sale_price_5y', np.nan),
            'remaining_balance_5y': row.get('remaining_balance_5y', np.nan),
            'net_sale_proceeds_5y': row.get('net_sale_proceeds_5y', np.nan),
            'payback_years_cashflow_only': row.get('payback_years_cashflow_only', None),
            'risk_flags': row.get('risk_flags', '')
        }

    def _generate_breakdown_html(self, prop_data: Dict) -> str:
        """Generate the breakdown HTML for a property using Python string formatting"""
        def format_currency(value):
            return f"${value:,.0f}"

        def format_percent(value):
            return f"{value:.2f}%"

        # Calculate conditional classes
        monthly_cf_class = 'positive' if prop_data['monthly_cash_flow'] >= 0 else 'negative'
        annual_cf_class = 'positive' if prop_data['annual_cash_flow'] >= 0 else 'negative'
        coc_class = 'positive' if prop_data['cash_on_cash_return'] >= 0 else 'warning'
        rent_coverage_class = 'positive' if prop_data['rent_coverage_ratio'] >= 1 else 'negative'
        one_percent_class = 'positive' if prop_data['one_percent_rule'] >= 1 else 'warning'

        # Calculate LTV ratio
        ltv_ratio = (prop_data['loan_amount'] / prop_data['price'] * 100) if prop_data['price'] > 0 else 0

        html = f"""
                <div class="calc-grid">
                    <div class="calc-section">
                        <h4>Property Overview</h4>
                        <div class="calc-line">
                            <span class="calc-label">Purchase Price:</span>
                            <span class="calc-value">{format_currency(prop_data['price'])}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Down Payment ({format_percent(prop_data['down_payment_percent'])}):</span>
                            <span class="calc-value">{format_currency(prop_data['down_payment'])}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Loan Amount:</span>
                            <span class="calc-value">{format_currency(prop_data['loan_amount'])}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Loan-to-Value Ratio:</span>
                            <span class="calc-value">{format_percent(ltv_ratio)}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Property Details:</span>
                            <span class="calc-value">{prop_data.get('beds', 'N/A')} • {prop_data.get('sqft', 'N/A')} sqft • {prop_data.get('neighbourhood', 'N/A')}</span>
                        </div>
                    </div>

                    <div class="calc-section">
                        <h4>Rental Income</h4>
                        <div class="calc-line">
                            <span class="calc-label">Estimated Monthly Rent:</span>
                            <span class="calc-value positive">{format_currency(prop_data['monthly_rent'])}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Effective Rent (95%):</span>
                            <span class="calc-value positive">{format_currency(prop_data['effective_monthly_rent'])}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Annual Income:</span>
                            <span class="calc-value positive">{format_currency(prop_data['annual_rental_income'])}</span>
                        </div>
                        <div class="calc-line highlight">
                            <span class="calc-label">Cap Rate:</span>
                            <span class="calc-value positive">{format_percent(prop_data['cap_rate'])}</span>
                        </div>
                    </div>

                    <div class="calc-section">
                        <h4>Monthly Expenses</h4>
                        <div class="calc-line">
                            <span class="calc-label">Mortgage (P&I):</span>
                            <span class="calc-value negative">{format_currency(prop_data['monthly_mortgage'])}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Property Tax:</span>
                            <span class="calc-value negative">{format_currency(prop_data['monthly_property_tax'])}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Insurance:</span>
                            <span class="calc-value negative">{format_currency(prop_data['monthly_insurance'])}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Property Mgmt (8%):</span>
                            <span class="calc-value negative">{format_currency(prop_data['monthly_management'])}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Maintenance Fee:</span>
                            <span class="calc-value negative">{format_currency(prop_data['monthly_maintenance'])}</span>
                        </div>
                        <div class="calc-line total">
                            <span class="calc-label">Total Expenses:</span>
                            <span class="calc-value negative">{format_currency(prop_data['total_monthly_expenses'])}</span>
                        </div>
                    </div>

                    <div class="calc-section">
                        <h4>Cash Flow Analysis</h4>
                        <div class="calc-line highlight">
                            <span class="calc-label">Monthly Cash Flow:</span>
                            <span class="calc-value {monthly_cf_class}">{format_currency(prop_data['monthly_cash_flow'])}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Annual Cash Flow:</span>
                            <span class="calc-value {annual_cf_class}">{format_currency(prop_data['annual_cash_flow'])}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Cash-on-Cash Return:</span>
                            <span class="calc-value {coc_class}">{format_percent(prop_data['cash_on_cash_return'])}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Break-Even Rent:</span>
                            <span class="calc-value">{format_currency(prop_data['break_even_rent'])}</span>
                        </div>
                    </div>

                    <div class="calc-section">
                        <h4>Investment Metrics</h4>
                        <div class="calc-line">
                            <span class="calc-label">Gross Rent Multiplier:</span>
                            <span class="calc-value">{prop_data['gross_rent_multiplier']:.2f}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Rent Coverage Ratio:</span>
                            <span class="calc-value {rent_coverage_class}">{prop_data['rent_coverage_ratio']:.2f}x</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">1% Rule:</span>
                            <span class="calc-value {one_percent_class}">{format_percent(prop_data['one_percent_rule'])}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Risk Score:</span>
                            <span class="calc-value">{prop_data['risk_score']:.1f} / 100</span>
                        </div>
                    </div>

                    <div class="calc-section">
                        <h4>ROI Projections</h4>
                        <div class="calc-line">
                            <span class="calc-label">Year 1 ROI:</span>
                            <span class="calc-value">{format_percent(prop_data['cash_on_cash_return'])}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">5-Year Projected ROI:</span>
                            <span class="calc-value positive">{format_percent(prop_data['cash_on_cash_return'] * 5 + 22.5)}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">10-Year Projected ROI:</span>
                            <span class="calc-value positive">{format_percent(prop_data['cash_on_cash_return'] * 10 + 45)}</span>
                        </div>
                    </div>
                </div>
        """
        return html
    
    def generate_html_report(self) -> str:
        """Generate comprehensive HTML report with dark mode, modern charts, and collapsible details"""
        summary = self.generate_executive_summary()
        strategies = self.analyzer.identify_investment_strategies(self.metrics_df)
        building_metrics = self.analyzer.market_intel.calculate_building_metrics()
        
        # Get top performers
        top_cashflow = self.metrics_df.nlargest(15, 'monthly_cash_flow')
        top_cap_rate = self.metrics_df.nlargest(15, 'cap_rate')
        top_overall = self.metrics_df.nlargest(40, 'cash_on_cash_return')
        
        # Prepare property data for JavaScript
        properties_data = []
        for idx, row in top_overall.iterrows():
            calc_data = self._generate_property_calculation_data(row)
            # Extract property details with proper formatting
            beds_display = str(row.get('beds_clean', 'N/A'))
            original_beds = str(row.get('beds', 'N/A'))

            if beds_display == 'N/A' or beds_display == '0' or not beds_display or beds_display == 'None':
                # Fallback to original beds column if beds_clean is not properly parsed
                original_beds_cleaned = original_beds.replace('bed', '').replace('BD', '').replace(' ', '').strip()
                if original_beds_cleaned and original_beds_cleaned != 'N/A' and original_beds_cleaned.lower() != 'none':
                    beds_display = original_beds_cleaned

            sqft_display = 'N/A'
            if pd.notna(row.get('sqft_clean')) and row.get('sqft_clean', 0) > 0:
                sqft_display = f"{row['sqft_clean']:.0f}"
            elif pd.notna(row.get('sqft')) and str(row.get('sqft', '')).replace('N/A', '').strip():
                # Fallback to original sqft if sqft_clean is not available
                sqft_value = DataParser.extract_number(str(row.get('sqft', 'N/A')))
                if sqft_value:
                    sqft_display = f"{sqft_value:.0f}"

            neighbourhood_display = str(row.get('neighbourhood', 'N/A'))
            if neighbourhood_display == 'N/A':
                neighbourhood_display = 'Toronto'  # Default fallback

            # Build full prop data for breakdown (ensure details are present)
            full_prop_data = {
                **calc_data,
                'beds': beds_display,
                'sqft': sqft_display,
                'neighbourhood': neighbourhood_display,
            }

            # Compute baths display
            baths_val = row.get('baths_clean', None)
            if pd.notna(baths_val) and baths_val is not None:
                try:
                    baths_num = float(baths_val)
                    baths_display = f"{int(baths_num)}" if abs(baths_num - int(baths_num)) < 1e-6 else f"{baths_num:.1f}"
                except Exception:
                    baths_display = 'N/A'
            else:
                baths_display = 'N/A'

            # Compute price per sqft display
            if pd.notna(row.get('sqft_clean')) and row.get('sqft_clean', 0) > 0:
                pps_value = row['price_numeric'] / row['sqft_clean']
                price_per_sqft_display = f"${pps_value:,.0f}"
            else:
                price_per_sqft_display = 'N/A'

            # Generate HTML breakdown using enriched property data
            breakdown_html = self._generate_breakdown_html(full_prop_data)

            # Days on market display
            dom_display = 'N/A'
            if pd.notna(row.get('days_on_market_clean')):
                try:
                    dom_val = int(row['days_on_market_clean'])
                    dom_display = f"{dom_val}d" if dom_val >= 1 else "<1d"
                except Exception:
                    dom_display = 'N/A'

            properties_data.append({
                'id': f'prop_{idx}',
                'address': str(row.get('address', 'N/A'))[:50],
                'neighbourhood': neighbourhood_display,
                'beds': beds_display,
                'sqft': sqft_display,
                'url': str(row.get('url', '#')),
                'breakdown_html': breakdown_html,
                'baths': baths_display,
                'price_per_sqft': price_per_sqft_display,
                'sqft_clean': row.get('sqft_clean', 600),
                'maintenance_monthly': row.get('monthly_maintenance', 300),
                'risk_category': row.get('risk_category', 'Medium Risk'),
                'total_risk_score': row.get('total_risk_score', 50),
                'days_on_market_display': dom_display,
                **calc_data
            })
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Toronto Condo Investment Analysis - Elite Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        :root {{
            --bg-primary: #0a0a0a;
            --bg-secondary: #111111;
            --bg-card: #1a1a1a;
            --bg-hover: #252525;
            --border: #2a2a2a;
            --text-primary: #ffffff;
            --text-secondary: #a0a0a0;
            --text-muted: #666666;
            --accent-primary: #00d4ff;
            --accent-secondary: #7c3aed;
            --accent-gradient: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%);
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --chart-1: #00d4ff;
            --chart-2: #7c3aed;
            --chart-3: #ec4899;
            --chart-4: #10b981;
            --chart-5: #f59e0b;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}
        
        .container {{
            max-width: 1920px;
            margin: 0 auto;
            background: var(--bg-primary);
        }}
        
        header {{
            background: var(--bg-secondary);
            padding: 4rem 3rem;
            text-align: center;
            position: relative;
            border-bottom: 1px solid var(--border);
            overflow: hidden;
        }}
        
        header::before {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(0, 212, 255, 0.1) 0%, transparent 70%);
            animation: rotate 30s linear infinite;
        }}
        
        @keyframes rotate {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}
        
        h1 {{
            font-size: 3.5rem;
            font-weight: 900;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1rem;
            position: relative;
            letter-spacing: -0.02em;
        }}
        
        .subtitle {{
            font-size: 1.1rem;
            color: var(--text-secondary);
            font-weight: 400;
            position: relative;
        }}
        
        .executive-summary {{
            padding: 3rem;
            background: var(--bg-primary);
            border-bottom: 1px solid var(--border);
        }}
        
        .section-title {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 2rem;
            background: linear-gradient(90deg, var(--text-primary) 0%, var(--text-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }}
        
        .metric-card {{
            background: var(--bg-card);
            padding: 2rem;
            border-radius: 16px;
            border: 1px solid var(--border);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }}
        
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background: var(--accent-gradient);
            transform: translateX(-100%);
            transition: transform 0.3s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-4px);
            background: var(--bg-hover);
            border-color: var(--accent-primary);
        }}
        
        .metric-card:hover::before {{
            transform: translateX(0);
        }}
        
        .metric-label {{
            font-size: 0.875rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.75rem;
            font-weight: 500;
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 800;
            color: var(--text-primary);
            line-height: 1;
            margin-bottom: 0.5rem;
        }}
        
        .metric-change {{
            font-size: 0.875rem;
            color: var(--text-muted);
        }}
        
        .positive {{ color: var(--success); }}
        .negative {{ color: var(--danger); }}
        .neutral {{ color: var(--warning); }}
        
        .charts-section {{
            padding: 3rem;
            background: var(--bg-primary);
        }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 2rem;
        }}
        
        .chart-card {{
            background: var(--bg-card);
            padding: 2rem;
            border-radius: 16px;
            border: 1px solid var(--border);
        }}
        
        .chart-container {{
            position: relative;
            height: 300px;
            width: 100%;
        }}
        
        .chart-title {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .data-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin-top: 2rem;
        }}
        
        .data-table thead {{
            background: var(--bg-card);
        }}
        
        .data-table th {{
            padding: 1rem;
            text-align: left;
            font-weight: 600;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border);
        }}
        
        .data-table tbody tr {{
            transition: background 0.2s;
            border-bottom: 1px solid var(--border);
        }}
        
        .data-table tbody tr:hover {{
            background: var(--bg-hover);
        }}
        
        .data-table tbody tr.clickable {{
            cursor: pointer;
        }}
        
        .data-table tbody tr.expanded {{
            background: var(--bg-hover);
        }}
        
        .data-table td {{
            padding: 1rem;
            color: var(--text-primary);
            border-bottom: 1px solid var(--border);
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.375rem 0.875rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .badge-success {{ 
            background: rgba(16, 185, 129, 0.1); 
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.2);
        }}
        .badge-warning {{ 
            background: rgba(245, 158, 11, 0.1); 
            color: var(--warning);
            border: 1px solid rgba(245, 158, 11, 0.2);
        }}
        .badge-danger {{ 
            background: rgba(239, 68, 68, 0.1); 
            color: var(--danger);
            border: 1px solid rgba(239, 68, 68, 0.2);
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
            transition: transform 0.3s;
        }}
        
        tr.expanded .expand-indicator {{
            transform: rotate(90deg);
        }}
        
        .click-hint {{
            color: #666;
            font-size: 0.85em;
            font-style: italic;
            margin-bottom: 10px;
        }}
        
        .footer {{
            background: var(--bg-secondary);
            color: var(--text-secondary);
            padding: 3rem;
            text-align: center;
            border-top: 1px solid var(--border);
        }}
        
        /* Modern scrollbar */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: var(--bg-secondary);
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: var(--border);
            border-radius: 5px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: var(--text-muted);
        }}
        
        @media (max-width: 768px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            
            h1 {{
                font-size: 2.5rem;
            }}
            
            .calc-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .top-properties {{
            padding: 3rem;
            background: var(--bg-secondary);
        }}
        
        .btn-link {{
            display: inline-block;
            padding: 0.5rem 0.875rem;
            background: var(--accent-gradient);
            color: #0b0b0b;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 700;
            text-decoration: none;
            transition: opacity 0.15s ease-in-out;
        }}
        .btn-link:hover {{
            opacity: 0.85;
        }}

        /* Configuration Panel Styles */
        .config-panel {{
            padding: 2rem;
            background: var(--bg-secondary);
            margin: 2rem 0;
            border-radius: 12px;
            border: 1px solid var(--border);
        }}

        .config-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}

        .config-toggle-btn {{
            padding: 0.5rem 1rem;
            background: var(--accent-gradient);
            color: #0b0b0b;
            border: none;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.15s ease-in-out;
        }}

        .config-toggle-btn:hover {{
            opacity: 0.85;
        }}

        .config-content {{
            margin-top: 1.5rem;
        }}

        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 2rem;
        }}

        .config-group {{
            background: var(--bg-card);
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}

        .config-group h3 {{
            margin: 0 0 1rem 0;
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--accent-primary);
        }}

        .config-item {{
            margin-bottom: 1rem;
        }}

        .config-item:last-child {{
            margin-bottom: 0;
        }}

        .config-item label {{
            display: block;
            margin-bottom: 0.5rem;
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-secondary);
        }}

        .config-item input {{
            width: 100%;
            padding: 0.5rem;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-primary);
            font-size: 0.875rem;
        }}

        .config-item input:focus {{
            outline: none;
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.1);
        }}

        .btn-primary, .btn-secondary {{
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 8px;
            font-size: 0.875rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease-in-out;
            margin-right: 0.5rem;
        }}

        .btn-primary {{
            background: var(--accent-gradient);
            color: #0b0b0b;
        }}

        .btn-primary:hover {{
            opacity: 0.85;
            transform: translateY(-1px);
        }}

        .btn-secondary {{
            background: var(--bg-primary);
            color: var(--text-primary);
            border: 1px solid var(--border);
        }}

        .btn-secondary:hover {{
            background: var(--bg-hover);
            border-color: var(--accent-primary);
        }}

        .recalc-status {{
            margin-top: 1rem;
            padding: 0.5rem;
            background: var(--bg-primary);
            border-radius: 6px;
            font-size: 0.875rem;
            color: var(--text-secondary);
            text-align: center;
        }}

        .recalc-status.updating {{
            background: rgba(245, 158, 11, 0.1);
            color: var(--warning);
            border: 1px solid var(--warning);
        }}

        .recalc-status.success {{
            background: rgba(16, 185, 129, 0.1);
            color: var(--success);
            border: 1px solid var(--success);
        }}

        @media (max-width: 768px) {{
            .config-grid {{
                grid-template-columns: 1fr;
            }}

            .config-header {{
                flex-direction: column;
                align-items: flex-start;
                gap: 1rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Toronto Condo Investment Analysis</h1>
            <p class="subtitle">Elite Market Intelligence Report • {datetime.now().strftime('%B %d, %Y')}</p>
            <button id="lang-toggle" class="config-toggle-btn" style="position:absolute; top:1rem; right:1rem;" onclick="toggleLanguage()">EN</button>
        </header>

        <section class="config-panel">
            <div class="config-header">
                <h2 class="section-title">Market Configuration</h2>
                <button id="config-toggle" class="config-toggle-btn" onclick="toggleConfigPanel()">Adjust Parameters</button>
            </div>

            <div id="config-content" class="config-content" style="display: none;">
                <div class="config-grid">
                    <div class="config-group">
                        <h3>Financing</h3>
                        <div class="config-item">
                            <label for="mortgage_rate">Mortgage Rate (%):</label>
                            <input type="number" id="mortgage_rate" step="0.1" value="{self.config.mortgage_rate}" onchange="updateConfig()">
                        </div>
                        <div class="config-item">
                            <label for="down_payment_percent">Down Payment (%):</label>
                            <input type="number" id="down_payment_percent" step="1" value="{self.config.down_payment_percent}" onchange="updateConfig()">
                        </div>
                    </div>

                    <div class="config-group">
                        <h3>Expenses</h3>
                        <div class="config-item">
                            <label for="property_tax_rate">Property Tax Rate (%):</label>
                            <input type="number" id="property_tax_rate" step="0.1" value="{self.config.property_tax_rate}" onchange="updateConfig()">
                        </div>
                        <div class="config-item">
                            <label for="vacancy_rate">Vacancy Rate (%):</label>
                            <input type="number" id="vacancy_rate" step="0.1" value="{self.config.vacancy_rate}" onchange="updateConfig()">
                        </div>
                        <div class="config-item">
                            <label for="insurance_monthly">Monthly Insurance ($):</label>
                            <input type="number" id="insurance_monthly" step="10" value="{self.config.insurance_monthly}" onchange="updateConfig()">
                        </div>
                        <div class="config-item">
                            <label for="property_management_rate">Management Rate (%):</label>
                            <input type="number" id="property_management_rate" step="0.1" value="{self.config.property_management_rate}" onchange="updateConfig()">
                        </div>
                    </div>

                    <div class="config-group">
                        <h3>Growth & Appreciation</h3>
                        <div class="config-item">
                            <label for="annual_appreciation">Annual Appreciation (%):</label>
                            <input type="number" id="annual_appreciation" step="0.1" value="{self.config.annual_appreciation}" onchange="updateConfig()">
                        </div>
                        <div class="config-item">
                            <label for="rent_growth_rate">Rent Growth Rate (%):</label>
                            <input type="number" id="rent_growth_rate" step="0.1" value="{self.config.rent_growth_rate}" onchange="updateConfig()">
                        </div>
                    </div>

                    <div class="config-group">
                        <h3 data-i18n="Actions">Actions</h3>
                        <button onclick="resetConfig()" class="btn-secondary" data-i18n="Reset to Default">Reset to Default</button>
                        <button onclick="recalculateAll()" class="btn-primary" data-i18n="Recalculate All">Recalculate All</button>
                        <div id="recalc-status" class="recalc-status" data-i18n="Ready to recalculate">Ready to recalculate</div>
                    </div>
                </div>
            </div>
        </section>

        <section class="executive-summary">
            <h2 class="section-title">Executive Summary</h2>
            
            <div class="summary-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Properties Analyzed</div>
                    <div class="metric-value">{summary['market_overview']['total_properties']:,}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Average Price</div>
                    <div class="metric-value">${summary['market_overview']['avg_price']/1000:.0f}K</div>
                    <div class="metric-change">Median: ${summary['market_overview']['median_price']/1000:.0f}K</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Average Cap Rate</div>
                    <div class="metric-value">{summary['investment_metrics']['avg_cap_rate']:.2f}%</div>
                    <div class="metric-change positive">Best: {summary['investment_metrics']['best_cap_rate']:.2f}%</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Positive Cash Flow</div>
                    <div class="metric-value">{summary['investment_metrics']['positive_cashflow_pct']:.1f}%</div>
                    <div class="metric-change">of properties</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Average Monthly Cash Flow</div>
                    <div class="metric-value">${summary['investment_metrics']['avg_monthly_cashflow']:.0f}</div>
                    <div class="metric-change positive">Best: ${summary['investment_metrics']['best_cashflow']:.0f}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Low Risk Properties</div>
                    <div class="metric-value">{summary['risk_profile']['low_risk_pct']:.1f}%</div>
                    <div class="metric-change">Safest: {summary['risk_profile']['safest_neighborhood']}</div>
                </div>
            </div>
        </section>
        
        <section class="charts-section">
            <h2 class="section-title">Market Analytics</h2>
            <div class="charts-grid">
                <div class="chart-card">
                    <h3 class="chart-title">Cash Flow Distribution</h3>
                    <div class="chart-container">
                        <canvas id="cashFlowChart"></canvas>
                    </div>
                </div>
                
                <div class="chart-card">
                    <h3 class="chart-title">Cap Rate Distribution</h3>
                    <div class="chart-container">
                        <canvas id="capRateChart"></canvas>
                    </div>
                </div>
                
                <div class="chart-card">
                    <h3 class="chart-title">Risk Assessment</h3>
                    <div class="chart-container">
                        <canvas id="riskChart"></canvas>
                    </div>
                </div>
                
                <div class="chart-card">
                    <h3 class="chart-title">Top Neighborhoods by Volume</h3>
                    <div class="chart-container">
                        <canvas id="neighborhoodChart"></canvas>
                    </div>
                </div>
            </div>
        </section>
        
        <section class="top-properties">
            <h2 class="section-title">Top Investment Opportunities</h2>
            <p class="click-hint">Click on any row to expand calculation details</p>
            
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Address</th>
                        <th>Price</th>
                        <th>Area</th>
                        <th>Sqft</th>
                        <th>Price/Sqft</th>
                        <th>Listed</th>
                        <th>Beds/Bath</th>
                        <th>Monthly CF</th>
                        <th>Cap Rate</th>
                        <th>CoC Return</th>
                        <th>Risk</th>
                        <th>DSCR</th>
                        <th>Link</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add top properties to table with collapsible details
        for idx, prop_data in enumerate(properties_data, 1):
            risk_badge = 'badge-success' if prop_data['risk_category'] == 'Low Risk' else 'badge-warning' if prop_data['risk_category'] == 'Medium Risk' else 'badge-danger'
            cf_class = 'positive' if prop_data['monthly_cash_flow'] > 0 else 'negative'
            
            # Format DSCR for table display
            dscr_val = prop_data.get('dscr', None)
            try:
                dscr_display = f"{dscr_val:.2f}x" if (dscr_val is not None and not pd.isna(dscr_val)) else 'N/A'
            except Exception:
                dscr_display = 'N/A'

            html_content += f"""
                    <tr class="clickable" data-prop-id="{prop_data['id']}" onclick="toggleBreakdown('{prop_data['id']}', event)">
                        <td>{idx}<span class="expand-indicator">▸</span></td>
                        <td>{prop_data['address']}</td>
                        <td>${prop_data['price']:,.0f}</td>
                        <td>{prop_data['neighbourhood']}</td>
                        <td>{prop_data['sqft']}</td>
                        <td>{prop_data['price_per_sqft']}</td>
                        <td>{prop_data.get('days_on_market_display', 'N/A')}</td>
                        <td>{prop_data['beds']}/{prop_data.get('baths', 'N/A')}</td>
                        <td class="{cf_class}">${prop_data['monthly_cash_flow']:.0f}</td>
                        <td>{prop_data['cap_rate']:.2f}%</td>
                        <td>{prop_data['cash_on_cash_return']:.1f}%</td>
                        <td><span class="badge {risk_badge}">{prop_data['risk_category']}</span></td>
                        <td>{dscr_display}</td>
                        <td><a href="{prop_data['url']}" target="_blank" rel="noopener noreferrer" class="btn-link">Open</a></td>
                    </tr>
                    <tr id="{prop_data['id']}_breakdown" style="display: none;">
                        <td colspan="14" style="padding: 0;">
                            <div class="calc-breakdown" id="{prop_data['id']}_content">
                                <!-- Content will be populated by JavaScript -->
                            </div>
                        </td>
                    </tr>
"""
        
        html_content += f"""
                </tbody>
            </table>
        </section>
        
        <footer class="footer">
            <p style="font-size: 1.1rem; margin-bottom: 1rem;">Elite Condo Investment Analysis System</p>
            <p style="opacity: 0.7;">Powered by Machine Learning • Market Comparables • Risk Analytics</p>
        </footer>
    </div>
    
    <script>
        // Store properties data
        const propertiesData = {json.dumps(properties_data)};

        // Store market configuration
        let marketConfig = {{
            mortgage_rate: {self.config.mortgage_rate},
            property_tax_rate: {self.config.property_tax_rate},
            down_payment_percent: {self.config.down_payment_percent},
            min_price: {self.config.min_price},
            vacancy_rate: {self.config.vacancy_rate},
            insurance_monthly: {self.config.insurance_monthly},
            property_management_rate: {self.config.property_management_rate},
            closing_costs_percent: {self.config.closing_costs_percent},
            annual_appreciation: {self.config.annual_appreciation},
            rent_growth_rate: {self.config.rent_growth_rate}
        }};

        // CRITICAL FIX #1: Store chart instances globally
        let chartInstances = {{}};

        // Language toggle (EN <-> 中文)
        let currentLang = 'zh';
        const i18nExact = {{
            'Market Configuration': '市场参数',
            'Adjust Parameters': '调整参数',
            'Hide Parameters': '隐藏参数',
            'Executive Summary': '执行摘要',
            'Market Analytics': '市场分析',
            'Top Investment Opportunities': '最佳投资机会',
            'Total Properties Analyzed': '分析的房源总数',
            'Average Price': '平均价格',
            'Average Cap Rate': '平均资本化率',
            'Positive Cash Flow': '正现金流',
            'Average Monthly Cash Flow': '平均月度现金流',
            'Low Risk Properties': '低风险房源',
            'Cash Flow Distribution': '现金流分布',
            'Cap Rate Distribution': '资本化率分布',
            'Risk Assessment': '风险评估',
            'Top Neighborhoods by Volume': '热门社区（按数量）',
            'Rank': '排名',
            'Address': '地址',
            'Price': '价格',
            'Area': '区域',
            'Sqft': '面积（平方英尺）',
            'Price/Sqft': '单价（每平方英尺）',
            'Listed': '挂牌天数',
            'Beds/Bath': '卧室/卫生间',
            'Monthly CF': '月度现金流',
            'Cap Rate': '资本化率',
            'CoC Return': '现金回报率',
            'Risk': '风险',
            'DSCR': '债务覆盖率',
            'Link': '链接',
            'Open': '打开',
            'Property Overview': '房产概览',
            'Purchase Price:': '购置价：',
            'Down Payment:': '首付：',
            'Loan Amount:': '贷款金额：',
            'Rental Income': '租金收入',
            'Estimated Monthly Rent:': '预计月租：',
            'Effective Rent:': '有效租金：',
            'Annual Income:': '年收入：',
            'Monthly Expenses': '月度支出',
            'Mortgage (P&I):': '按揭（月供）：',
            'Property Tax:': '地产税：',
            'Insurance:': '保险：',
            'Property Mgmt': '物业管理',
            'Maintenance Fee:': '物业费：',
            'Total Expenses:': '总支出：',
            'Cash Flow Analysis': '现金流分析',
            'Monthly Cash Flow:': '月度现金流：',
            'Annual Cash Flow:': '年度现金流：',
            'Cash-on-Cash Return:': '现金回报率：',
            'Break-Even Rent:': '盈亏平衡租金：',
            'Investment Metrics': '投资指标',
            'Gross Rent Multiplier:': '总租金倍数：',
            'Rent Coverage Ratio:': '租金覆盖率：',
            '1% Rule:': '1% 法则：',
            'ROI Projections': '收益预测',
            'Debt & Risk': '债务与风险',
            'DSCR:': 'DSCR：',
            'DSCR (+100bp):': 'DSCR（+100bp）：',
            'Monthly CF (+100bp):': '月度现金流（+100bp）：',
            'Risk Flags:': '风险标记：',
            'Max Loan for DSCR=1.0:': 'DSCR=1.0 可承贷上限：',
            'Suggested Down Payment (DSCR 1.0):': '建议首付（DSCR 1.0）：',
            'Down Payment Gap (to DSCR 1.0):': '与DSCR 1.0的首付缺口：',
            'Unit Metrics': '户型指标',
            'Price/Sqft:': '单价/平方英尺：',
            'Rent/Sqft:': '租金/平方英尺：',
            'PPSF Z-Score:': '单价Z分数：',
            'Break-Even Occupancy:': '盈亏平衡出租率：',
            'Principal & ROE': '本金与回报',
            'Principal Paid (Year 1):': '第一年偿还本金：',
            'ROE (Year 1, incl. Principal):': '第一年权益回报（含本金）：',
            'Stress Tests': '压力测试',
            'DSCR (Rent -10%):': 'DSCR（租金-10%）：',
            'Monthly CF (Rent -10%):': '月度现金流（租金-10%）：',
            'DSCR (Vacancy 10%):': 'DSCR（空置率10%）：',
            'Monthly CF (Vacancy 10%):': '月度现金流（空置10%）：',
            'DSCR (Maintenance +20%):': 'DSCR（物业费+20%）：',
            'Monthly CF (Maintenance +20%):': '月度现金流（物业费+20%）：',
            '5-Year Projection': '5年预测',
            'IRR (5-Year):': '内部收益率（5年）：',
            'NPV (5-Year):': '净现值（5年）：',
            'Sale Price (Year 5):': '第5年售价：',
            'Remaining Balance (Year 5):': '第5年剩余贷款：',
            'Net Sale Proceeds (Year 5):': '第5年售后净得：',
            'Payback (Years, CF only):': '回收期（仅现金流，年）：',
            'Year 1 ROI:': '第1年投资回报：',
            '5-Year Projected ROI:': '5年预测回报：',
            '10-Year Projected ROI:': '10年预测回报：',
            'Low Risk': '低风险',
            'Medium Risk': '中等风险',
            'High Risk': '高风险',
            'Very High Risk': '极高风险',
            'Financing': '融资',
            'Expenses': '支出',
            'Growth & Appreciation': '增长与升值',
            'Actions': '操作',
            'Mortgage Rate (%):': '按揭利率（%）：',
            'Down Payment (%):': '首付比例（%）：',
            'Property Tax Rate (%):': '地产税率（%）：',
            'Vacancy Rate (%):': '空置率（%）：',
            'Monthly Insurance ($):': '月度保险（$）：',
            'Management Rate (%):': '物业管理费率（%）：',
            'Annual Appreciation (%):': '年升值（%）：',
            'Rent Growth Rate (%):': '租金增长率（%）：',
            'Reset to Default': '重置为默认',
            'Recalculate All': '重新计算',
            'Ready to recalculate': '可重新计算',
            'Parameters updated - Click "Recalculate All" to apply changes': '参数已更新 - 点击"重新计算"应用更改',
            'Configuration reset to defaults': '配置已重置为默认值',
            'Recalculating investment metrics...': '正在重新计算投资指标...',
            'All calculations updated successfully!': '所有计算已成功更新！',
            'Error:': '错误：',
            'Click on any row to expand calculation details': '点击任意行展开计算明细'
        }};

        const i18nPartial = {{
            'Elite Market Intelligence Report': '精英市场情报报告',
            'Median:': '中位数：',
            'Best:': '最佳：',
            'Safest:': '最安全：',
            'of properties': '的房源',
            'Property Mgmt (': '物业管理（',
            'Effective Rent (': '有效租金（',
            'Down Payment (': '首付（'
        }};

        function translateText(text) {{
            if (!text) return text;
            const trimmed = text.trim();
            if (i18nExact[trimmed]) return i18nExact[trimmed];
            let out = text;
            Object.keys(i18nPartial).forEach(key => {{
                if (out.includes(key)) out = out.replaceAll(key, i18nPartial[key]);
            }});
            return out;
        }}

        function applyTranslations() {{
            // Translate main title via data attribute baseline
            const titleEl = document.querySelector('header h1');
            if (titleEl) {{
                if (!titleEl.dataset.i18nOriginal) titleEl.dataset.i18nOriginal = titleEl.textContent;
                // Provide explicit mapping for the title
                const titleEn = titleEl.dataset.i18nOriginal;
                const titleZh = '多伦多公寓投资分析';
                titleEl.textContent = (currentLang === 'zh') ? titleZh : titleEn;
            }}
            // Config toggle button text uses English baseline, then translates if needed
            const cfgBtn = document.getElementById('config-toggle');
            const cfgContent = document.getElementById('config-content');
            if (cfgBtn) {{
                const english = (cfgContent && cfgContent.style.display === 'none') ? 'Adjust Parameters' : 'Hide Parameters';
                cfgBtn.textContent = (currentLang === 'zh') ? translateText(english) : english;
            }}

            // Elements to translate; cache original English in data-i18n-original
            const selectors = ['.section-title', '.chart-title', '.metric-label', '.click-hint', 'th', 'label', '.config-group h3', '.btn-link', '.calc-label', '.metric-change', '.calc-section h4', '.subtitle', '.badge'];
            selectors.forEach(sel => {{
                document.querySelectorAll(sel).forEach(el => {{
                    if (!el.dataset.i18nOriginal) {{
                        el.dataset.i18nOriginal = el.textContent;
                    }}
                    el.textContent = (currentLang === 'zh') ? translateText(el.dataset.i18nOriginal) : el.dataset.i18nOriginal;
                }});
            }});

            // Also translate any element that explicitly declares a data-i18n key (e.g., buttons, status labels)
            document.querySelectorAll('[data-i18n]').forEach(el => {{
                const key = el.getAttribute('data-i18n');
                if (!el.dataset.i18nOriginal) {{
                    el.dataset.i18nOriginal = key || el.textContent;
                }}
                const base = el.dataset.i18nOriginal || el.textContent;
                el.textContent = (currentLang === 'zh') ? translateText(base) : base;
            }});

            // Update risk chart labels from cached English base
            if (chartInstances.riskChart) {{
                if (!window.baseRiskLabels) window.baseRiskLabels = [...chartInstances.riskChart.data.labels];
                chartInstances.riskChart.data.labels = (currentLang === 'zh') ? window.baseRiskLabels.map(label => translateText(label)) : window.baseRiskLabels;
                chartInstances.riskChart.update('none');
            }}
        }}

        function toggleLanguage() {{
            currentLang = currentLang === 'en' ? 'zh' : 'en';
            const btn = document.getElementById('lang-toggle');
            if (btn) btn.textContent = currentLang === 'zh' ? 'EN' : '中文';
            applyTranslations();
        }}

        // Ensure default Chinese is applied on load
        document.addEventListener('DOMContentLoaded', () => {{
            const btn = document.getElementById('lang-toggle');
            if (btn) btn.textContent = currentLang === 'zh' ? 'EN' : '中文';
            applyTranslations();
        }});

        // Helper function for distribution counts
        function getCounts(data, bins) {{
            const counts = Array(bins.length - 1).fill(0);
            data.forEach(value => {{
                for (let i = 0; i < bins.length - 1; i++) {{
                    if (value >= bins[i] && value < bins[i + 1]) {{
                        counts[i]++;
                        break;
                    }}
                }}
            }});
            return counts;
        }}

        // Format functions
        function formatCurrency(value) {{
            return new Intl.NumberFormat('en-US', {{
                style: 'currency',
                currency: 'USD',
                minimumFractionDigits: 0,
                maximumFractionDigits: 0
            }}).format(value);
        }}

        function formatPercent(value) {{
            return value.toFixed(2) + '%';
        }}
        
        // CRITICAL FIX #2: Generate breakdown HTML dynamically (not using pre-generated)
        function generateBreakdownHTML(prop) {{
            // Recalculate all dynamic values based on current config
            const down_payment_percent = (prop.down_payment / prop.price * 100) || 0;
            const ltv_ratio = (prop.loan_amount / prop.price * 100) || 0;
            const monthly_cf_class = prop.monthly_cash_flow >= 0 ? 'positive' : 'negative';
            const annual_cf_class = prop.annual_cash_flow >= 0 ? 'positive' : 'negative';
            const coc_class = prop.cash_on_cash_return >= 0 ? 'positive' : 'warning';
            const rent_coverage_class = prop.rent_coverage_ratio >= 1 ? 'positive' : 'negative';
            const one_percent_class = prop.one_percent_rule >= 1 ? 'positive' : 'warning';

            return `
                <div class="calc-grid">
                    <div class="calc-section">
                        <h4>Property Overview</h4>
                        <div class="calc-line">
                            <span class="calc-label">Purchase Price:</span>
                            <span class="calc-value">${{formatCurrency(prop.price)}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Down Payment (${{formatPercent(down_payment_percent)}}):</span>
                            <span class="calc-value">${{formatCurrency(prop.down_payment)}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Loan Amount:</span>
                            <span class="calc-value">${{formatCurrency(prop.loan_amount)}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Loan-to-Value Ratio:</span>
                            <span class="calc-value">${{formatPercent(ltv_ratio)}}</span>
                        </div>
                    </div>

                    <div class="calc-section">
                        <h4>Rental Income</h4>
                        <div class="calc-line">
                            <span class="calc-label">Estimated Monthly Rent:</span>
                            <span class="calc-value positive">${{formatCurrency(prop.monthly_rent)}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Effective Rent (${{formatPercent(100 - marketConfig.vacancy_rate)}}):</span>
                            <span class="calc-value positive">${{formatCurrency(prop.effective_monthly_rent)}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Annual Income:</span>
                            <span class="calc-value positive">${{formatCurrency(prop.annual_rental_income)}}</span>
                        </div>
                        <div class="calc-line highlight">
                            <span class="calc-label">Cap Rate:</span>
                            <span class="calc-value positive">${{formatPercent(prop.cap_rate)}}</span>
                        </div>
                    </div>

                    <div class="calc-section">
                        <h4>Monthly Expenses</h4>
                        <div class="calc-line">
                            <span class="calc-label">Mortgage (P&I):</span>
                            <span class="calc-value negative">${{formatCurrency(prop.monthly_mortgage)}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Property Tax:</span>
                            <span class="calc-value negative">${{formatCurrency(prop.monthly_property_tax)}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Insurance:</span>
                            <span class="calc-value negative">${{formatCurrency(prop.monthly_insurance)}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Property Mgmt (${{formatPercent(marketConfig.property_management_rate)}}):</span>
                            <span class="calc-value negative">${{formatCurrency(prop.monthly_management)}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Maintenance Fee:</span>
                            <span class="calc-value negative">${{formatCurrency(prop.monthly_maintenance)}}</span>
                        </div>
                        <div class="calc-line total">
                            <span class="calc-label">Total Expenses:</span>
                            <span class="calc-value negative">${{formatCurrency(prop.total_monthly_expenses)}}</span>
                        </div>
                    </div>

                    <div class="calc-section">
                        <h4>Cash Flow Analysis</h4>
                        <div class="calc-line highlight">
                            <span class="calc-label">Monthly Cash Flow:</span>
                            <span class="calc-value ${{monthly_cf_class}}">${{formatCurrency(prop.monthly_cash_flow)}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Annual Cash Flow:</span>
                            <span class="calc-value ${{annual_cf_class}}">${{formatCurrency(prop.annual_cash_flow)}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Cash-on-Cash Return:</span>
                            <span class="calc-value ${{coc_class}}">${{formatPercent(prop.cash_on_cash_return)}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Break-Even Rent:</span>
                            <span class="calc-value">${{formatCurrency(prop.break_even_rent)}}</span>
                        </div>
                    </div>

                    <div class="calc-section">
                        <h4>Investment Metrics</h4>
                        <div class="calc-line">
                            <span class="calc-label">Gross Rent Multiplier:</span>
                            <span class="calc-value">${{prop.gross_rent_multiplier.toFixed(2)}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Rent Coverage Ratio:</span>
                            <span class="calc-value ${{rent_coverage_class}}">${{prop.rent_coverage_ratio.toFixed(2)}}x</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">1% Rule:</span>
                            <span class="calc-value ${{one_percent_class}}">${{formatPercent(prop.one_percent_rule)}}</span>
                        </div>
                    </div>
                    <div class="calc-section">
                        <h4>Debt & Risk</h4>
                        <div class="calc-line">
                            <span class="calc-label">DSCR:</span>
                            <span class="calc-value">${{(isFinite(prop.dscr) ? prop.dscr.toFixed(2) + 'x' : 'N/A')}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">DSCR (+100bp):</span>
                            <span class="calc-value">${{(isFinite(prop.dscr_up_100bp) ? prop.dscr_up_100bp.toFixed(2) + 'x' : 'N/A')}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Monthly CF (+100bp):</span>
                            <span class="calc-value ${{prop.monthly_cash_flow_up_100bp>=0?'positive':'negative'}}">${{(isFinite(prop.monthly_cash_flow_up_100bp) ? formatCurrency(prop.monthly_cash_flow_up_100bp) : 'N/A')}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Risk Flags:</span>
                            <span class="calc-value">${{(prop.risk_flags && prop.risk_flags.length>0) ? prop.risk_flags : 'None'}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Max Loan for DSCR=1.0:</span>
                            <span class="calc-value">${{(isFinite(prop.max_loan_for_dscr1) ? formatCurrency(prop.max_loan_for_dscr1) : 'N/A')}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Suggested Down Payment (DSCR 1.0):</span>
                            <span class="calc-value">${{(isFinite(prop.suggested_down_payment_dscr1) ? formatCurrency(prop.suggested_down_payment_dscr1) : 'N/A')}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Down Payment Gap (to DSCR 1.0):</span>
                            <span class="calc-value ${{prop.down_payment_gap_dscr1>0?'warning':''}}">${{(isFinite(prop.down_payment_gap_dscr1) ? formatCurrency(prop.down_payment_gap_dscr1) : 'N/A')}}</span>
                        </div>
                    </div>

                    <div class="calc-section">
                        <h4>Unit Metrics</h4>
                        <div class="calc-line">
                            <span class="calc-label">Price/Sqft:</span>
                            <span class="calc-value">${{(isFinite(prop.price_per_sqft) ? '$' + Math.round(prop.price_per_sqft) : 'N/A')}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Rent/Sqft:</span>
                            <span class="calc-value">${{(isFinite(prop.rent_per_sqft) ? '$' + prop.rent_per_sqft.toFixed(2) : 'N/A')}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">PPSF Z-Score:</span>
                            <span class="calc-value">${{(isFinite(prop.ppsf_zscore_neighborhood) ? prop.ppsf_zscore_neighborhood.toFixed(2) : 'N/A')}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Break-Even Occupancy:</span>
                            <span class="calc-value">${{(isFinite(prop.breakeven_occupancy) ? (prop.breakeven_occupancy*100).toFixed(1) + '%' : 'N/A')}}</span>
                        </div>
                    </div>

                    <div class="calc-section">
                        <h4>Principal & ROE</h4>
                        <div class="calc-line">
                            <span class="calc-label">Principal Paid (Year 1):</span>
                            <span class="calc-value">${{(isFinite(prop.principal_paid_year1) ? formatCurrency(prop.principal_paid_year1) : 'N/A')}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">ROE (Year 1, incl. Principal):</span>
                            <span class="calc-value">${{(isFinite(prop.roe_year1_total) ? prop.roe_year1_total.toFixed(2) + '%' : 'N/A')}}</span>
                        </div>
                    </div>

                    <div class="calc-section">
                        <h4>Stress Tests</h4>
                        <div class="calc-line">
                            <span class="calc-label">DSCR (Rent -10%):</span>
                            <span class="calc-value">${{(isFinite(prop.dscr_rent_minus10) ? prop.dscr_rent_minus10.toFixed(2) + 'x' : 'N/A')}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Monthly CF (Rent -10%):</span>
                            <span class="calc-value ${{prop.monthly_cash_flow_rent_minus10>=0?'positive':'negative'}}">${{(isFinite(prop.monthly_cash_flow_rent_minus10) ? formatCurrency(prop.monthly_cash_flow_rent_minus10) : 'N/A')}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">DSCR (Vacancy 10%):</span>
                            <span class="calc-value">${{(isFinite(prop.dscr_vacancy_10) ? prop.dscr_vacancy_10.toFixed(2) + 'x' : 'N/A')}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Monthly CF (Vacancy 10%):</span>
                            <span class="calc-value ${{prop.monthly_cash_flow_vacancy_10>=0?'positive':'negative'}}">${{(isFinite(prop.monthly_cash_flow_vacancy_10) ? formatCurrency(prop.monthly_cash_flow_vacancy_10) : 'N/A')}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">DSCR (Maintenance +20%):</span>
                            <span class="calc-value">${{(isFinite(prop.dscr_maint_plus20) ? prop.dscr_maint_plus20.toFixed(2) + 'x' : 'N/A')}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Monthly CF (Maintenance +20%):</span>
                            <span class="calc-value ${{prop.monthly_cash_flow_maint_plus20>=0?'positive':'negative'}}">${{(isFinite(prop.monthly_cash_flow_maint_plus20) ? formatCurrency(prop.monthly_cash_flow_maint_plus20) : 'N/A')}}</span>
                        </div>
                    </div>

                    <div class="calc-section">
                        <h4>5-Year Projection</h4>
                        <div class="calc-line">
                            <span class="calc-label">IRR (5-Year):</span>
                            <span class="calc-value">${{(prop.irr_5y!=null ? (Number(prop.irr_5y).toFixed(2) + '%') : 'N/A')}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">NPV (5-Year):</span>
                            <span class="calc-value">${{(isFinite(prop.npv_5y) ? formatCurrency(prop.npv_5y) : 'N/A')}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Sale Price (Year 5):</span>
                            <span class="calc-value">${{(isFinite(prop.sale_price_5y) ? formatCurrency(prop.sale_price_5y) : 'N/A')}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Remaining Balance (Year 5):</span>
                            <span class="calc-value">${{(isFinite(prop.remaining_balance_5y) ? formatCurrency(prop.remaining_balance_5y) : 'N/A')}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Net Sale Proceeds (Year 5):</span>
                            <span class="calc-value">${{(isFinite(prop.net_sale_proceeds_5y) ? formatCurrency(prop.net_sale_proceeds_5y) : 'N/A')}}</span>
                        </div>
                        <div class="calc-line">
                            <span class="calc-label">Payback (Years, CF only):</span>
                            <span class="calc-value">${{(prop.payback_years_cashflow_only!=null ? prop.payback_years_cashflow_only : 'N/A')}}</span>
                        </div>
                    </div>
                </div>
            `;
        }}

        // Toggle breakdown - now uses dynamic generation
        function toggleBreakdown(propId, event) {{
            const breakdownRow = document.getElementById(propId + '_breakdown');
            const contentDiv = document.getElementById(propId + '_content');
            const clickedRow = event.currentTarget;

            if (breakdownRow.style.display === 'none') {{
                const propData = propertiesData.find(p => p.id === propId);
                if (propData) {{
                    // Always regenerate with current values
                    contentDiv.innerHTML = generateBreakdownHTML(propData);
                    contentDiv.classList.add('show');
                    // Ensure translations apply to newly injected content
                    if (typeof applyTranslations === 'function') {{
                        applyTranslations();
                    }}
                }}
                breakdownRow.style.display = 'table-row';
                clickedRow.classList.add('expanded');
            }} else {{
                breakdownRow.style.display = 'none';
                contentDiv.classList.remove('show');
                clickedRow.classList.remove('expanded');
            }}
        }}
        
        // CRITICAL FIX #7: Initialize charts and store references
        Chart.defaults.color = '#888';
        Chart.defaults.borderColor = '#333';

        // Cash Flow Distribution Chart - Store reference in chartInstances
        const cashFlowData = {json.dumps(self.metrics_df['monthly_cash_flow'].dropna().tolist())};
        const cashFlowBins = [-2000, -1500, -1000, -500, 0, 500, 1000, 1500, 2000];
        const cashFlowCounts = getCounts(cashFlowData, cashFlowBins);

        chartInstances.cashFlowChart = new Chart(document.getElementById('cashFlowChart'), {{
            type: 'bar',
            data: {{
                labels: cashFlowBins.slice(0, -1).map((v, i) => `${{v}} to $${{cashFlowBins[i+1]}}`),
                datasets: [{{
                    label: 'Properties',
                    data: cashFlowCounts,
                    backgroundColor: 'rgba(0, 212, 255, 0.6)',
                    borderColor: 'rgba(0, 212, 255, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                animation: {{ duration: 0 }},
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        ticks: {{ color: '#888' }},
                        grid: {{ color: '#2a2a2a' }}
                    }},
                    x: {{
                        ticks: {{
                            color: '#888',
                            maxRotation: 45,
                            minRotation: 45
                        }},
                        grid: {{ color: '#2a2a2a' }}
                    }}
                }}
            }}
        }});

        // Cap Rate Distribution Chart - Store reference in chartInstances
        const capRateData = {json.dumps(self.metrics_df['cap_rate'].dropna().tolist())};
        const capRateBins = [0, 2, 4, 6, 8, 10, 12];
        const capRateCounts = getCounts(capRateData, capRateBins);

        chartInstances.capRateChart = new Chart(document.getElementById('capRateChart'), {{
            type: 'bar',
            data: {{
                labels: capRateBins.slice(0, -1).map((v, i) => `${{v}}% to $${{capRateBins[i+1]}}%`),
                datasets: [{{
                    label: 'Properties',
                    data: capRateCounts,
                    backgroundColor: 'rgba(124, 58, 237, 0.6)',
                    borderColor: 'rgba(124, 58, 237, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                animation: {{ duration: 0 }},
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        ticks: {{ color: '#888' }},
                        grid: {{ color: '#2a2a2a' }}
                    }},
                    x: {{
                        ticks: {{ color: '#888' }},
                        grid: {{ color: '#2a2a2a' }}
                    }}
                }}
            }}
        }});

        // Risk Assessment Chart
        const riskCounts = {json.dumps(self.metrics_df['risk_category'].value_counts().to_dict())};

        chartInstances.riskChart = new Chart(document.getElementById('riskChart'), {{
            type: 'doughnut',
            data: {{
                labels: Object.keys(riskCounts),
                datasets: [{{
                    data: Object.values(riskCounts),
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.6)',
                        'rgba(245, 158, 11, 0.6)',
                        'rgba(239, 68, 68, 0.6)',
                        'rgba(124, 58, 237, 0.6)'
                    ],
                    borderColor: [
                        'rgba(16, 185, 129, 1)',
                        'rgba(245, 158, 11, 1)',
                        'rgba(239, 68, 68, 1)',
                        'rgba(124, 58, 237, 1)'
                    ],
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'right',
                        labels: {{
                            color: '#888'
                        }}
                    }}
                }}
            }}
        }});

        // Top Neighborhoods Chart
        const neighborhoodData = {json.dumps(
            self.metrics_df['neighbourhood'].value_counts().head(10).to_dict()
        )};

        chartInstances.neighborhoodChart = new Chart(document.getElementById('neighborhoodChart'), {{
            type: 'bar',
            data: {{
                labels: Object.keys(neighborhoodData),
                datasets: [{{
                    label: 'Properties',
                    data: Object.values(neighborhoodData),
                    backgroundColor: 'rgba(236, 72, 153, 0.6)',
                    borderColor: 'rgba(236, 72, 153, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    x: {{
                        beginAtZero: true,
                        ticks: {{ color: '#888' }},
                        grid: {{ color: '#2a2a2a' }}
                    }},
                    y: {{
                        ticks: {{ color: '#888' }},
                        grid: {{ color: '#2a2a2a' }}
                    }}
                }}
            }}
        }});

        // Config panel functions
        function toggleConfigPanel() {{
            const content = document.getElementById('config-content');
            const button = document.getElementById('config-toggle');

            if (content.style.display === 'none') {{
                content.style.display = 'block';
                button.innerHTML = translateText('Hide Parameters');
            }} else {{
                content.style.display = 'none';
                button.innerHTML = translateText('Adjust Parameters');
            }}
        }}

        function updateConfig() {{
            marketConfig.mortgage_rate = parseFloat(document.getElementById('mortgage_rate').value);
            marketConfig.property_tax_rate = parseFloat(document.getElementById('property_tax_rate').value);
            marketConfig.down_payment_percent = parseFloat(document.getElementById('down_payment_percent').value);
            marketConfig.vacancy_rate = parseFloat(document.getElementById('vacancy_rate').value);
            marketConfig.insurance_monthly = parseFloat(document.getElementById('insurance_monthly').value);
            marketConfig.property_management_rate = parseFloat(document.getElementById('property_management_rate').value);
            marketConfig.annual_appreciation = parseFloat(document.getElementById('annual_appreciation').value);
            marketConfig.rent_growth_rate = parseFloat(document.getElementById('rent_growth_rate').value);

            const status = document.getElementById('recalc-status');
            status.textContent = translateText('Parameters updated - Click "Recalculate All" to apply changes');
            status.className = 'recalc-status';
        }}

        function resetConfig() {{
            document.getElementById('mortgage_rate').value = 5.5;
            document.getElementById('property_tax_rate').value = 0.7;
            document.getElementById('down_payment_percent').value = 20;
            document.getElementById('vacancy_rate').value = 5;
            document.getElementById('insurance_monthly').value = 100;
            document.getElementById('property_management_rate').value = 8;
            document.getElementById('annual_appreciation').value = 4.5;
            document.getElementById('rent_growth_rate').value = 3.5;

            updateConfig();

            const status = document.getElementById('recalc-status');
            status.textContent = translateText('Configuration reset to defaults');
            status.className = 'recalc-status success';

            setTimeout(() => {{
                status.textContent = translateText('Ready to recalculate');
                status.className = 'recalc-status';
            }}, 2000);
        }}

        // CRITICAL FIX #3: Properly recalculate all metrics
        function recalculatePropertyMetrics(prop) {{
            const price = prop.price;
            const monthlyRent = prop.monthly_rent;
            const maintenanceFee = prop.maintenance_monthly || 300;

            // Recalculate all financial metrics
            prop.down_payment = price * (marketConfig.down_payment_percent / 100);
            prop.loan_amount = price - prop.down_payment;

            // Mortgage calculation
            const r = marketConfig.mortgage_rate / 100 / 12;
            const n = 30 * 12;
            if (r > 0) {{
                prop.monthly_mortgage = prop.loan_amount * (r * Math.pow(1 + r, n)) / (Math.pow(1 + r, n) - 1);
            }} else {{
                prop.monthly_mortgage = prop.loan_amount / n;
            }}

            // Monthly expenses
            prop.monthly_property_tax = price * (marketConfig.property_tax_rate / 100 / 12);
            prop.monthly_insurance = marketConfig.insurance_monthly;
            prop.monthly_management = monthlyRent * (marketConfig.property_management_rate / 100);
            prop.monthly_maintenance = maintenanceFee;

            prop.total_monthly_expenses = prop.monthly_mortgage + prop.monthly_property_tax +
                                              prop.monthly_insurance + prop.monthly_management + maintenanceFee;

            // Income calculations
            prop.effective_monthly_rent = monthlyRent * (1 - marketConfig.vacancy_rate / 100);
            prop.annual_rental_income = prop.effective_monthly_rent * 12;

            // Cash flow
            prop.monthly_cash_flow = prop.effective_monthly_rent - prop.total_monthly_expenses;
            prop.annual_cash_flow = prop.monthly_cash_flow * 12;

            // Investment returns
            prop.cap_rate = ((prop.annual_rental_income -
                                (maintenanceFee + prop.monthly_property_tax + prop.monthly_insurance) * 12) /
                                price * 100) || 0;
            prop.cash_on_cash_return = (prop.down_payment > 0 ?
                                           prop.annual_cash_flow / prop.down_payment * 100 : 0);

            // Additional metrics
            prop.gross_rent_multiplier = prop.annual_rental_income > 0 ?
                                            price / prop.annual_rental_income : 0;
            prop.break_even_rent = prop.total_monthly_expenses / (1 - marketConfig.vacancy_rate / 100);
            prop.rent_coverage_ratio = prop.break_even_rent > 0 ?
                                          monthlyRent / prop.break_even_rent : 0;
            prop.one_percent_rule = monthlyRent / price * 100;
        }}

        // Main recalculation function
        function recalculateAll() {{
            const status = document.getElementById('recalc-status');
            status.textContent = translateText('Recalculating investment metrics...');
            status.className = 'recalc-status updating';

            setTimeout(() => {{
                try {{
                    // Recalculate all properties
                    propertiesData.forEach(function(prop) {{
                        recalculatePropertyMetrics(prop);
                    }});

                    // Update displays
                    updateTableRows();
                    updateSummaryStats();
                    updateAllCharts();

                    // Clear any open breakdowns to force refresh
                    document.querySelectorAll('.calc-breakdown.show').forEach(div => {{
                        div.classList.remove('show');
                        div.innerHTML = '';
                    }});

                    status.textContent = translateText('All calculations updated successfully!');
                    status.className = 'recalc-status success';

                    setTimeout(() => {{
                        status.textContent = translateText('Ready to recalculate');
                        status.className = 'recalc-status';
                    }}, 3000);

                }} catch (error) {{
                    console.error('Recalculation error:', error);
                    status.textContent = translateText('Error:') + ' ' + error.message;
                    status.className = 'recalc-status updating';
                }}
            }}, 500);
        }}

        // CRITICAL FIX #4: Update table rows properly
        function updateTableRows() {{
            propertiesData.forEach(function(prop) {{
                const row = document.querySelector('tr[data-prop-id="' + prop.id + '"]');
                if (row) {{
                    const cells = row.querySelectorAll('td');

                    // Update Monthly CF (column index 8)
                    if (cells[8]) {{
                        cells[8].className = prop.monthly_cash_flow >= 0 ? 'positive' : 'negative';
                        cells[8].textContent = '$' + Math.round(prop.monthly_cash_flow);
                    }}

                    // Update Cap Rate (column index 9)
                    if (cells[9]) {{
                        cells[9].textContent = prop.cap_rate.toFixed(2) + '%';
                    }}

                    // Update CoC Return (column index 10)
                    if (cells[10]) {{
                        cells[10].textContent = prop.cash_on_cash_return.toFixed(1) + '%';
                    }}

                    // Update Price/Sqft (column index 5)
                    if (cells[5]) {{
                        const ppsf = parseFloat(prop.price_per_sqft);
                        if (isFinite(ppsf)) {{
                            cells[5].textContent = '$' + ppsf;
                        }} else {{
                            cells[5].textContent = 'N/A';
                        }}
                    }}
                }}
            }});
        }}

        // CRITICAL FIX #5: Update summary statistics properly
        function updateSummaryStats() {{
            const totalProperties = propertiesData.length;
            if (totalProperties === 0) return;

            // Calculate statistics
            const avgPrice = propertiesData.reduce((sum, p) => sum + p.price, 0) / totalProperties;
            const sortedPrices = propertiesData.map(p => p.price).sort((a, b) => a - b);
            const medianPrice = sortedPrices[Math.floor(totalProperties / 2)];

            const avgCapRate = propertiesData.reduce((sum, p) => sum + p.cap_rate, 0) / totalProperties;
            const bestCapRate = Math.max(...propertiesData.map(p => p.cap_rate));

            const positiveCashflowCount = propertiesData.filter(p => p.monthly_cash_flow > 0).length;
            const positiveCashflowPct = (positiveCashflowCount / totalProperties) * 100;

            const avgMonthlyCashflow = propertiesData.reduce((sum, p) => sum + p.monthly_cash_flow, 0) / totalProperties;
            const bestCashflow = Math.max(...propertiesData.map(p => p.monthly_cash_flow));

            // Update metric cards
            const metricCards = document.querySelectorAll('.metric-card');

            // Card 2: Average Price
            if (metricCards[1]) {{
                const valueDiv = metricCards[1].querySelector('.metric-value');
                const changeDiv = metricCards[1].querySelector('.metric-change');
                if (valueDiv) valueDiv.textContent = '$' + Math.round(avgPrice / 1000) + 'K';
                if (changeDiv) changeDiv.textContent = 'Median: $' + Math.round(medianPrice / 1000) + 'K';
            }}

            // Card 3: Average Cap Rate
            if (metricCards[2]) {{
                const valueDiv = metricCards[2].querySelector('.metric-value');
                const changeDiv = metricCards[2].querySelector('.metric-change');
                if (valueDiv) valueDiv.textContent = avgCapRate.toFixed(2) + '%';
                if (changeDiv) {{
                    changeDiv.textContent = 'Best: ' + bestCapRate.toFixed(2) + '%';
                    changeDiv.className = 'metric-change positive';
                }}
            }}

            // Card 4: Positive Cash Flow
            if (metricCards[3]) {{
                const valueDiv = metricCards[3].querySelector('.metric-value');
                if (valueDiv) valueDiv.textContent = positiveCashflowPct.toFixed(1) + '%';
            }}

            // Card 5: Average Monthly Cash Flow
            if (metricCards[4]) {{
                const valueDiv = metricCards[4].querySelector('.metric-value');
                const changeDiv = metricCards[4].querySelector('.metric-change');
                if (valueDiv) valueDiv.textContent = '$' + Math.round(avgMonthlyCashflow);
                if (changeDiv) {{
                    changeDiv.textContent = 'Best: $' + Math.round(bestCashflow);
                    changeDiv.className = 'metric-change positive';
                }}
            }}
        }}

        // CRITICAL FIX #6: Update charts with new data
        function updateAllCharts() {{
            // Update Cash Flow Chart
            if (chartInstances.cashFlowChart) {{
                const cashFlowData = propertiesData.map(p => p.monthly_cash_flow);
                const cashFlowBins = [-2000, -1500, -1000, -500, 0, 500, 1000, 1500, 2000];
                const cashFlowCounts = getCounts(cashFlowData, cashFlowBins);

                chartInstances.cashFlowChart.data.datasets[0].data = cashFlowCounts;
                chartInstances.cashFlowChart.update('none');
            }}

            // Update Cap Rate Chart
            if (chartInstances.capRateChart) {{
                const capRateData = propertiesData.map(p => p.cap_rate);
                const capRateBins = [0, 2, 4, 6, 8, 10, 12];
                const capRateCounts = getCounts(capRateData, capRateBins);

                chartInstances.capRateChart.data.datasets[0].data = capRateCounts;
                chartInstances.capRateChart.update('none');
            }}
        }}


    </script>
</body>
</html>
"""
        
        # Save report to data directory
        os.makedirs('data', exist_ok=True)
        filename = f'toronto_condo_analysis_enhanced_{self.timestamp}.html'
        filepath = os.path.join('data', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Enhanced HTML report generated: {filepath}")
        return filepath
    



class EnhancedCondoInvestmentPipeline:
    """Main pipeline orchestrating the entire analysis"""
    
    def __init__(self, csv_file: str, config: Optional[MarketConfig] = None):
        # Check if csv_file is just a filename or a path
        if not os.path.dirname(csv_file):
            # If no directory specified, look in data folder
            csv_file = os.path.join('data', csv_file)
        self.csv_file = csv_file
        self.config = config or MarketConfig()
        self.df = None
        self.metrics_df = None
        self.analyzer = None
    
    def load_and_clean_data(self) -> pd.DataFrame:
        """Load and perform initial data cleaning"""
        print("\n" + "="*80)
        print("🏢 TORONTO CONDO INVESTMENT ANALYZER - PROFESSIONAL EDITION")
        print("="*80 + "\n")
        
        print(f"📂 Loading data from: {self.csv_file}")
        self.df = pd.read_csv(self.csv_file)
        print(f"✅ Loaded {len(self.df)} properties")
        
        # Clean numeric fields
        self.df['price_numeric'] = self.df['price'].apply(DataParser.extract_number)
        
        # Filter by minimum price
        self.df = self.df[
            (self.df['price_numeric'].notna()) & 
            (self.df['price_numeric'] >= self.config.min_price)
        ].copy()
        print(f"📊 After filtering: {len(self.df)} properties (≥${self.config.min_price:,.0f})")
        
        # Parse structured fields
        print("\n🔧 Parsing property features...")
        
        # Bedroom configuration
        self.df['bedroom_config'] = self.df['beds'].apply(DataParser.parse_bedroom_config)
        self.df['bedroom_numeric'] = self.df['bedroom_config'].apply(lambda x: x.get('numeric', 0))
        self.df['beds_clean'] = self.df['bedroom_config'].apply(
            lambda x: f"{x.get('beds', 0)}+{x.get('den_rooms', 0)}" if x.get('den') and x.get('beds') is not None else
                     ('Studio' if x.get('beds') == 0 else str(x.get('beds', 'N/A')))
        )

                # Other fields
        self.df['baths_clean'] = self.df['baths'].apply(lambda x: DataParser.extract_number(x) or 0)
        self.df['parking_clean'] = self.df['parking'].apply(lambda x: DataParser.extract_number(x) or 0)
        self.df['sqft_clean'] = self.df['sqft'].apply(DataParser.extract_number)
        self.df['maintenance_monthly'] = self.df['maintenance_fees'].apply(lambda x: DataParser.extract_number(x) or 0)
        self.df['days_on_market_clean'] = self.df['days_on_market'].apply(DataParser.parse_time_period)
        # Prefer DOM from listing_date/scraped_date when available
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            listing_dt = pd.to_datetime(self.df.get('listing_date'), errors='coerce')
            scraped_dt = pd.to_datetime(self.df.get('scraped_date'), errors='coerce')
            dom_from_dates = (scraped_dt - listing_dt).dt.days
        self.df['days_on_market_clean'] = np.where(
            pd.notna(dom_from_dates), dom_from_dates, self.df['days_on_market_clean']
        )
        
        # Parse taxes
        self.df['annual_taxes'] = self.df['taxes'].apply(DataParser.extract_number)
        
        print(f"✅ Data cleaning complete")
        
        return self.df
    
    def run_analysis(self) -> Tuple[pd.DataFrame, Dict]:
        """Run complete investment analysis"""
        print("\n🚀 Running comprehensive investment analysis...")
        
        # Initialize analyzer
        self.analyzer = InvestmentAnalyzer(self.df, self.config)
        
        # Extract market intelligence
        print("\n📈 Extracting market intelligence...")
        price_histories = self.analyzer.market_intel.extract_price_histories()
        comparables = self.analyzer.market_intel.analyze_comparable_properties()
        
        # Add market data to dataframe
        for idx in price_histories:
            if 'metrics' in price_histories[idx]:
                for metric, value in price_histories[idx]['metrics'].items():
                    # Handle tuple/list values by converting to string or taking first value
                    if isinstance(value, (list, tuple)):
                        if metric == 'rent_range' and len(value) == 2:
                            self.df.loc[idx, f'hist_rent_min'] = value[0]
                            self.df.loc[idx, f'hist_rent_max'] = value[1]
                        else:
                            self.df.loc[idx, f'hist_{metric}'] = str(value)
                    else:
                        self.df.loc[idx, f'hist_{metric}'] = value
        
        for idx in comparables:
            if 'metrics' in comparables[idx]:
                for metric, value in comparables[idx]['metrics'].items():
                    # Handle tuple/list values
                    if isinstance(value, (list, tuple)):
                        if metric == 'rent_range' and len(value) == 2:
                            self.df.loc[idx, 'rent_range_min'] = value[0]
                            self.df.loc[idx, 'rent_range_max'] = value[1]
                        else:
                            self.df.loc[idx, metric] = str(value)
                    else:
                        self.df.loc[idx, metric] = value
        
        # Calculate investment metrics
        print("\n💰 Calculating investment metrics...")
        self.metrics_df = self.analyzer.calculate_advanced_metrics()
        
        # Risk assessment
        print("\n⚠️ Performing risk assessment...")
        self.metrics_df = self.analyzer.calculate_risk_scores(self.metrics_df)
        
        # Identify strategies
        print("\n🎯 Identifying investment strategies...")
        strategies = self.analyzer.identify_investment_strategies(self.metrics_df)
        
        # Print strategy summary
        print("\n📋 Investment Strategies Summary:")
        for strategy_name, strategy_df in strategies.items():
            if len(strategy_df) > 0:
                print(f"   • {strategy_name.replace('_', ' ').title()}: {len(strategy_df)} properties")
                if len(strategy_df) > 0:
                    best = strategy_df.iloc[0]
                    print(f"     Best: {best['address'][:40] if pd.notna(best['address']) else 'N/A'} - ${best['price_numeric']:,.0f}")
        
        return self.metrics_df, strategies
    
    def generate_reports(self) -> str:
        """Generate HTML report"""
        print("\n📝 Generating HTML report...")

        # Initialize report generator
        report_gen = ReportGenerator(self.analyzer, self.metrics_df, self.config)

        # Generate HTML report
        html_file = report_gen.generate_html_report()

        return html_file
    
    def run(self) -> Dict:
        """Run complete analysis pipeline"""
        try:
            # Load and clean data
            self.load_and_clean_data()
            
            # Run analysis
            self.metrics_df, strategies = self.run_analysis()
            
            # Generate HTML report
            html_file = self.generate_reports()
            
            # Summary statistics
            print("\n" + "="*80)
            print("✅ ANALYSIS COMPLETE!")
            print("="*80)
            
            # Key findings
            positive_cf = (self.metrics_df['monthly_cash_flow'] > 0).sum()
            avg_cap = self.metrics_df['cap_rate'].mean()
            best_cap = self.metrics_df['cap_rate'].max()
            best_cf = self.metrics_df.loc[self.metrics_df['monthly_cash_flow'].idxmax()]
            
            print(f"\n🎯 Key Findings:")
            print(f"   • Properties with positive cash flow: {positive_cf}/{len(self.metrics_df)} ({positive_cf/len(self.metrics_df)*100:.1f}%)")
            print(f"   • Average cap rate: {avg_cap:.2f}%")
            print(f"   • Best cap rate: {best_cap:.2f}%")
            print(f"   • Best cash flow property: ${best_cf['monthly_cash_flow']:.0f}/month")
            print(f"     {best_cf['address'][:50] if pd.notna(best_cf['address']) else 'N/A'}")
            
            print(f"\n📊 Report Generated:")
            print(f"   • HTML Report: {html_file}")
            print(f"\n💡 Open the HTML report in your browser for interactive visualizations and collapsible calculation details")

            return {
                'success': True,
                'html_report': html_file,
                'metrics_df': self.metrics_df,
                'strategies': strategies
            }
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}


def main():
    """Main execution function"""
    # Configuration - use the actual CSV file from the project
    CSV_FILE = 'toronto_condos_robust_20250910_124956.csv'  # Updated to match actual file
    
    # Market parameters for 2025 Toronto
    config = MarketConfig(
        mortgage_rate=3.5,
        property_tax_rate=0.5,
        down_payment_percent=30,
        min_price=300000,
        vacancy_rate=5,
        insurance_monthly=0,
        property_management_rate=0,
        closing_costs_percent=2,
        annual_appreciation=3.5,
        rent_growth_rate=2.5
    )
    
    # Run pipeline
    pipeline = EnhancedCondoInvestmentPipeline(CSV_FILE, config)
    results = pipeline.run()
    
    return results


if __name__ == "__main__":
    main()
