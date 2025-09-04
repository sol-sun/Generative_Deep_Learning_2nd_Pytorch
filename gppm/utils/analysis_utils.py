"""
分析用ユーティリティ関数
"""

from typing import List, Dict
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gppm.providers.rbics_provider import RBICSDataProvider


def explore_specific_category(category_companies: Dict[str, pd.DataFrame], category_name: str, top_n: int = 20) -> pd.DataFrame:
    if category_name not in category_companies:
        print(f"カテゴリー '{category_name}' が見つかりません。")
        print("利用可能なカテゴリー:")
        for i, cat in enumerate(sorted(category_companies.keys())[:10]):
            print(f"  {i+1}. {cat}")
        return pd.DataFrame()

    df = category_companies[category_name].head(top_n)

    print(f"=== {category_name} カテゴリー詳細 ===")
    print(f"該当企業数: {len(category_companies[category_name])}")
    print(f"表示企業数: {min(top_n, len(df))}")

    if 'FF_SALES' in df.columns:
        print(f"総売上高: {df['FF_SALES'].sum():,.0f}")
        print(f"平均売上高: {df['FF_SALES'].mean():,.0f}")
        print(f"平均依存度: {df['CATEGORY_VALUE'].mean():.3f}")

    print(f"\n上位企業:")
    display_cols = ['FF_CO_NAME', 'CATEGORY_VALUE', 'FF_SALES', 'CURRENCY', 'FTERM_2']
    available_cols = [col for col in display_cols if col in df.columns]
    print(df[available_cols].to_string(index=False))

    return df


def compare_categories(summary_df: pd.DataFrame, category_companies: Dict[str, pd.DataFrame], categories: List[str]) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=('企業数', '総売上高', '平均依存度', '最大依存度'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}], [{'type': 'bar'}, {'type': 'bar'}]],
    )

    filtered_summary = summary_df[summary_df['Category'].isin(categories)]
    if len(filtered_summary) == 0:
        print("指定されたカテゴリーが見つかりません")
        return None

    fig.add_trace(go.Bar(x=filtered_summary['Category'], y=filtered_summary['Company_Count'], name='企業数'), row=1, col=1)

    if 'Total_Sales' in filtered_summary.columns:
        fig.add_trace(go.Bar(x=filtered_summary['Category'], y=filtered_summary['Total_Sales'], name='総売上高'), row=1, col=2)

    if 'Avg_Dependency' in filtered_summary.columns:
        fig.add_trace(go.Bar(x=filtered_summary['Category'], y=filtered_summary['Avg_Dependency'], name='平均依存度'), row=2, col=1)

    if 'Max_Dependency' in filtered_summary.columns:
        fig.add_trace(go.Bar(x=filtered_summary['Category'], y=filtered_summary['Max_Dependency'], name='最大依存度'), row=2, col=2)

    fig.update_layout(height=600, showlegend=False, title_text="カテゴリー比較分析")
    return fig


def get_cross_category_companies(category_companies: Dict[str, pd.DataFrame], categories: List[str], min_dependency: float = 0.05) -> pd.DataFrame:
    cross_companies = []
    company_category_map: Dict[str, Dict] = {}

    for category in categories:
        if category in category_companies:
            high_dep_companies = category_companies[category][category_companies[category]['CATEGORY_VALUE'] >= min_dependency]
            for _, company in high_dep_companies.iterrows():
                entity_id = company['FACTSET_ENTITY_ID']
                if entity_id not in company_category_map:
                    company_category_map[entity_id] = {'FF_CO_NAME': company['FF_CO_NAME'], 'categories': [], 'dependencies': []}
                company_category_map[entity_id]['categories'].append(category)
                company_category_map[entity_id]['dependencies'].append(company['CATEGORY_VALUE'])

    for entity_id, info in company_category_map.items():
        if len(info['categories']) > 1:
            cross_companies.append({
                'FACTSET_ENTITY_ID': entity_id,
                'FF_CO_NAME': info['FF_CO_NAME'],
                'Categories': ', '.join(info['categories']),
                'Category_Count': len(info['categories']),
                'Max_Dependency': max(info['dependencies']),
                'Total_Dependency': sum(info['dependencies']),
            })

    if cross_companies:
        result_df = pd.DataFrame(cross_companies)
        result_df = result_df.sort_values('Total_Dependency', ascending=False)
        return result_df
    else:
        return pd.DataFrame()


def get_specific_company_data(entity_id: str = None):
    from gppm.providers.factset_provider import FactSetProvider

    financial_provider = FactSetProvider()
    # 修正：新しいAPIに対応
    if entity_id:
        identity_records = financial_provider.get_identity_records(
            entity_id=entity_id
        )
    else:
        identity_records = financial_provider.get_identity_records()
    
    # レコードをDataFrameに変換
    entity_data = []
    for record in identity_records:
        entity_data.append({
            'FACTSET_ENTITY_ID': record.factset_entity_id,
            'FSYM_ID': record.fsym_id,
            'FF_CO_NAME': record.company_name,
            'ISO_COUNTRY_FACT': record.headquarters_country_code,
            'PRIMARY_EQUITY_FLAG': 1 if record.is_primary_equity else 0
        })
    
    import pandas as pd
    return pd.DataFrame(entity_data)


def clean_geographic_data(text_list: List[str]):
    from gppm.finance.geographic_processor import GeographicProcessor

    geo_processor = GeographicProcessor()
    return [geo_processor.remove_geographic_info(text) for text in text_list]


def get_rbics_master():
    rbics_provider = RBICSDataProvider()
    return rbics_provider.get_master_table()


