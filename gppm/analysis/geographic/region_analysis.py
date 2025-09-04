"""
地理的情報処理クラス

テキストから地理的情報の除去、通貨から地域マッピングなどの機能を提供
"""

from typing import List
import pandas as pd


class GeographicProcessor:
    """地理的情報の処理クラス"""
    
    def __init__(self):
        self.geographic_regions = [
            "Americas (Excluding US)", "Mideast/Africa", "Africa and Non-US Americas", "Middle East and Africa",
            "Australia/New Zealand", "Other Asia/Pac", "Canada", "Latin America", "Caribbean",
            "Australia and New Zealand", "Other Asia/Pacific", "China", "Africa",
            "Europe", "Eastern Europe", "Northern Europe", "Southern Europe", "Western Europe", "Middle East",
            "Pan-EMEA", "Multinational", "Other United States", "US Territories and Puerto Rico",
            "United States Southern", "United States South Central", "United States Northeast",
            "United States South Atlantic", "United States Atlantic", "United States Pacific",
            "United States West South Central", "United States Western", "Middle Atlantic US",
            "New England United States", "North East United States", "Northwest United States",
            "South United States", "Western United States", "Diversified United States", "United States",
            "Other Americas (Excluding US)", "Australian", "Pan-Asia/Pacific", "Eastern Region",
            "Other US Western", "Pan-US", "US Rockies", "Gulf Coast", "US Mexico", "Mid-Continent",
            "Other US South", "Permian Basin", "Australia/NZ", "North Asia", "North Sea", "Rest of Europe",
            "Pan-Europe", "Other Europe", "MENA", "Russia/CIS/FSU", "Russia/CIS", "Sub-Saharan",
            "Mixed International", "International", "Asia-Europe", "Central and South America", "Mexico",
            "Asia Excluding China", "Americas", "Other Americas", "Global", "Other International",
            "Other Asia/Pacific", "Europe, Middle East and Africa", "United States", "Multinational",
            "Americas", "Asia/Pacific", "US and Canada", "Pan-Americas", "US", "Asia (Excluding China)",
            "Australia and New Zealand", "China", "Pan-Asia/Pacific", "Central and Eastern Europe",
            "Western Europe", "Multi-Region", "Multi-Type United States", "Other United States", "Canada",
            "Latin America", "Other Americas", "Other North America", "Australian", "Pan-Asia", "Rest of Asia",
            "Southeast Asia", "Other Africa", "Russia and CIS", "South Africa", "Central and South America",
            "Pan-Europe", "Middle East/Africa", "Australia including Oceania", "Other Asia/Pacific",
            "Diversified", "Australia/Oceania", "Rest of Asia/Pac", "Europe, Middle East and Africa",
            "United States", "Pan-EMEA", "Asia/Pacific"
        ]
        # 長いものから順にソート（より具体的な地域名を優先）
        self.geographic_regions = sorted(self.geographic_regions, key=len, reverse=True)
    
    def remove_geographic_info(self, text: str) -> str:
        """テキストから地理的情報を除去"""
        for region in self.geographic_regions:
            if text.startswith(region):
                return text[len(region):].lstrip()
        return text
    
    def get_region_mapping(self, currencies: List[str]) -> pd.DataFrame:
        """通貨から地域へのマッピングを取得
        
        簡略化された通貨-地域マッピング実装
        """
        # 主要通貨の地域マッピング辞書
        currency_to_region = {
            'USD': 'Americas',
            'CAD': 'Americas',
            'MXN': 'Americas',
            'BRL': 'Americas',
            'EUR': 'Europe',
            'GBP': 'Europe',
            'CHF': 'Europe',
            'SEK': 'Europe',
            'NOK': 'Europe',
            'JPY': 'Asia/Pacific',
            'CNY': 'Asia/Pacific',
            'HKD': 'Asia/Pacific',
            'SGD': 'Asia/Pacific',
            'KRW': 'Asia/Pacific',
            'AUD': 'Asia/Pacific',
            'NZD': 'Asia/Pacific',
            'INR': 'Asia/Pacific',
            'ZAR': 'Africa',
            'AED': 'Middle East',
            'SAR': 'Middle East'
        }
        
        regions = []
        for currency in currencies:
            region = currency_to_region.get(currency, 'Other')
            regions.append(region)
        
        return pd.DataFrame({
            "REGION": regions,
            "CURRENCY": currencies
        })


