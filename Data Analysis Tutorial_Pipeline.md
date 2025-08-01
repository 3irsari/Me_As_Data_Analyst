"""
This script demonstrates the end-to-end process of data analysis from data generation
to insights visualization. In a real-world scenario, data would typically come from
SQL databases, but for demonstration purposes, we'll generate dummy data and work
with CSV files.

Author: Birce SARI
Date: 2025
Purpose: Suggested demonstration of complete analysis pipeline
"""

# =============================================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime,timedelta
import random
import warnings
import os

# Statistical libraries
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12,8)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("COMPLETE DATA ANALYSIS PIPELINE")
print("=" * 80)
print("Libraries imported successfully!")
print()


# =============================================================================
# STEP 2: DATA GENERATION (Simulating Database Export)
# =============================================================================

def generate_sales_data(num_records=1000, seed=42) :
    """
    Generate dummy sales data that simulates real e-commerce transactions.

    In a real scenario, this data would come from SQL queries like:

    SELECT
        s.sale_id,
        s.sale_date,
        p.product_name,
        r.region_name,
        s.channel,
        s.quantity,
        s.unit_price,
        c.customer_age,
        s.customer_satisfaction_score,
        s.marketing_spend,
        s.discount_applied
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
    JOIN regions r ON s.region_id = r.region_id
    JOIN customers c ON s.customer_id = c.customer_id
    WHERE s.sale_date >= '2024-01-01'
        AND s.sale_date < '2025-01-01'
        AND s.status = 'Completed'
    ORDER BY s.sale_date;

    IN SUMMARRY:
    🗓️ Date Generation: Capped at 365 days
    📦 Product Selection & Pricing: Random product chosen from a predefined list. Price range is
    fetched from a dictionary (product_price_ranges). Unit price is sampled within that product’s
    price range.
    🔢 Quantity of Items: 80% of orders are small (1–3 items). 20% are bulk (4–15 items).
    🌍 Region Selection: Higher weights mean more activity in predetrmined areas.
    🛍️ Sales Channel: Online is the most common (45%).
    👤 Customer Age: Ages follow a normal distribution, centered around 38. Clipped to legal
    shopping age (18–75).
    😊 Customer Satisfaction Score: Base score: 3.5 out of 5. High-priced items → slightly better score.
    Online sales → slightly worse due to shipping issues. Final score drawn from normal distribution,
    clipped to 1–5.
    💰 Marketing Spend: Modeled with exponential distribution to simulate variability. Online and Direct
     channels spend more.
    🎁 Discount Applied: 30% of sales have a random discount (5–25%).
    🧮 Revenue Calculations: Discount: Applied if relevant. Net revenue: Revenue after discount.
    🧪 Data Quality Issues: 5% of records are missing satisfaction scores. 1% are extreme outliers
    (e.g., 100 items).
    📄 Assemble Record: A dictionary of the generated data is added to a list. Sale ID is formatted
    like SAL-000001.
    🧾 Create DataFrame and Summary: Data is converted to a Pandas DataFrame and Returned a clean,
    realistic synthetic sales DataFrame

    After each step, related information is provided for the user about the status.

    """

    print("\nSTEP 2: GENERATING DUMMY DATA")
    print("-" * 40)

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Define business parameters (realistic distributions)
    products = ['Laptop','Smartphone','Headphones','Tablet','Smart Watch','Camera','Speaker','Monitor']
    regions = ['North America','Europe','Asia Pacific','Latin America','Middle East & Africa']
    channels = ['Online','Retail Store','Partner Channel','Direct Sales']

    # Product pricing ranges (simulating different product categories)
    product_price_ranges = {
        'Laptop' : (800,2500),
        'Smartphone' : (300,1200),
        'Headphones' : (50,400),
        'Tablet' : (200,800),
        'Smart Watch' : (150,600),
        'Camera' : (400,2000),
        'Speaker' : (100,500),
        'Monitor' : (200,1000)
    }

    data = []

    print(f"Generating {num_records} sales records...")

    for i in range(num_records) :
        # Generate date (weighted towards recent months)
        days_back = np.random.exponential(scale=60)  # More recent dates more likely
        days_back = min(days_back,365)  # Cap at 1 year
        sale_date = datetime.now() - timedelta(days=int(days_back))

        # Select product and determine price based on product category
        product = random.choice(products)
        price_range = product_price_ranges[product]
        unit_price = round(np.random.uniform(price_range[0],price_range[1]),2)

        # Generate quantity (most orders are 1-3 items, some bulk orders)
        if random.random() < 0.8 :  # 80% are small orders
            quantity = random.randint(1,3)
        else :  # 20% are larger orders
            quantity = random.randint(4,15)

        # Region selection (some regions more active)
        region_weights = [0.35,0.25,0.20,0.12,0.08]  # NA, EU, APAC, LATAM, MEA
        region = np.random.choice(regions,p=region_weights)

        # Channel selection (online growing)
        channel_weights = [0.45,0.30,0.15,0.10]  # Online, Retail, Partner, Direct
        channel = np.random.choice(channels,p=channel_weights)

        # Customer demographics
        customer_age = int(np.random.normal(loc=38,scale=12))  # Normal distribution around 38
        customer_age = max(18,min(75,customer_age))  # Bound between 18-75

        # Satisfaction score (influenced by product quality and price)
        base_satisfaction = 3.5
        if unit_price > np.mean(price_range) :  # Higher priced items tend to have better satisfaction
            base_satisfaction += 0.3
        if channel == 'Online' :  # Online has slightly lower satisfaction due to shipping issues
            base_satisfaction -= 0.2

        satisfaction = round(np.random.normal(loc=base_satisfaction,scale=0.8),1)
        satisfaction = max(1.0,min(5.0,satisfaction))  # Bound between 1-5

        # Marketing spend (varies by channel and region)
        marketing_multiplier = {'Online' : 25,'Retail Store' : 15,'Partner Channel' : 10,'Direct Sales' : 30}
        base_marketing = marketing_multiplier[channel]
        marketing_spend = round(np.random.exponential(scale=base_marketing),2)

        # Discount applied (promotional campaigns)
        if random.random() < 0.3 :  # 30% of sales have discounts
            discount_percent = round(np.random.uniform(5,25),1)
        else :
            discount_percent = 0.0

        # Calculate final metrics
        gross_revenue = quantity * unit_price
        discount_amount = gross_revenue * (discount_percent / 100)
        net_revenue = gross_revenue - discount_amount

        # Add some realistic data quality issues (like real-world data)
        # Occasionally missing satisfaction scores
        if random.random() < 0.05 :  # 5% missing satisfaction
            satisfaction = np.nan

        # Occasionally extreme outliers
        if random.random() < 0.01 :  # 1% outliers
            quantity = random.randint(50,100)  # Bulk orders

        data.append({
            'sale_id' : f'SAL-{i + 1:06d}',
            'sale_date' : sale_date.strftime('%Y-%m-%d'),
            'product_name' : product,
            'region_name' : region,
            'channel' : channel,
            'quantity' : quantity,
            'unit_price' : unit_price,
            'customer_age' : customer_age,
            'customer_satisfaction_score' : satisfaction,
            'marketing_spend' : marketing_spend,
            'discount_percent' : discount_percent,
            'gross_revenue' : gross_revenue,
            'discount_amount' : discount_amount,
            'net_revenue' : net_revenue
        })

    df = pd.DataFrame(data)

    print(f"✓ Generated {len(df)} sales records")
    print(f"✓ Date range: {df['sale_date'].min()} to {df['sale_date'].max()}")
    print(f"✓ Products: {', '.join(df['product_name'].unique())}")
    print(f"✓ Regions: {', '.join(df['region_name'].unique())}")
    print()

    return df


# Generate the data
sales_data = generate_sales_data(1000)


# =============================================================================
# STEP 3: DATA EXPORT TO CSV (Simulating Database Export)
# =============================================================================

def export_to_csv(dataframe,filename='sales_data.csv') :
    """
    Export dataframe to CSV file.

    In real scenarios, data would be exported from databases using connections like:

    df = pd.read_sql(query, connection)
    df.to_csv('sales_data.csv', index=False)
    """

    print("\nSTEP 3: EXPORTING DATA TO CSV")
    print("-" * 40)

    try :
        dataframe.to_csv(filename,index=False)
        print(f"✓ Data exported to {filename}")
        print(f"✓ File size: {os.path.getsize(filename) / 1024:.1f} KB")
        print()
    except Exception as e :
        print(f"✗ Error exporting data: {e}")
        print()


# Export the generated data
export_to_csv(sales_data)


# =============================================================================
# STEP 4: DATA LOADING AND INITIAL INSPECTION
# =============================================================================

def load_and_inspect_data(filename='sales_data.csv') :
    """
    Load data from CSV and perform initial inspection.
    This simulates loading data from various sources in real scenarios.
    """

    print("\nSTEP 4: DATA LOADING AND INITIAL INSPECTION")
    print("-" * 40)

    # Load data
    try :
        df = pd.read_csv(filename)
        print(f"✓ Data loaded successfully from {filename}")
    except Exception as e :
        print(f"✗ Error loading data: {e}")
        return None

    # Basic information
    print(f"\nDATASET OVERVIEW:")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    # Data types
    print(f"\nDATA TYPES:")
    for col,dtype in df.dtypes.items() :
        print(f"  {col}: {dtype}")

    # Missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0 :
        print(f"\nMISSING VALUES:")
        for col,count in missing_counts[missing_counts > 0].items() :
            percentage = (count / len(df)) * 100
            print(f"  {col}: {count} ({percentage:.1f}%)")
    else :
        print(f"\n✓ No missing values detected")

    # Basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\nNUMERIC COLUMNS SUMMARY:")
    print(df[numeric_cols].describe().round(2))

    print()
    return df


# Load the data
df = load_and_inspect_data()


# =============================================================================
# STEP 5: DATA CLEANING AND PREPARATION
# =============================================================================

def clean_and_prepare_data(dataframe) :
    """
    Clean and prepare data for analysis.
    This includes handling missing values, outliers, and creating derived columns.
    """

    print("\nSTEP 5: DATA CLEANING AND PREPARATION")
    print("-" * 40)

    df_clean = dataframe.copy()

    # Convert date column to datetime
    print("Converting date columns...")
    df_clean['sale_date'] = pd.to_datetime(df_clean['sale_date'])

    # Create time-based features
    df_clean['year'] = df_clean['sale_date'].dt.year
    df_clean['month'] = df_clean['sale_date'].dt.month
    df_clean['quarter'] = df_clean['sale_date'].dt.quarter
    df_clean['day_of_week'] = df_clean['sale_date'].dt.day_name()
    df_clean['is_weekend'] = df_clean['sale_date'].dt.weekday >= 5

    # Handle missing satisfaction scores
    missing_satisfaction = df_clean['customer_satisfaction_score'].isnull().sum()
    if missing_satisfaction > 0 :
        print(f"Handling {missing_satisfaction} missing satisfaction scores...")
        # Fill with median by product (business logic: similar products have similar satisfaction)
        df_clean['customer_satisfaction_score'] = df_clean.groupby('product_name')[
            'customer_satisfaction_score'].transform(
            lambda x : x.fillna(x.median())
        )

    # Create customer age groups
    df_clean['age_group'] = pd.cut(df_clean['customer_age'],
                                   bins=[0,25,35,45,55,100],
                                   labels=['18-25','26-35','36-45','46-55','55+'])

    # Create revenue per unit
    df_clean['revenue_per_unit'] = df_clean['net_revenue'] / df_clean['quantity']

    # Create profit margin (assuming 60% of revenue is profit after costs)
    df_clean['estimated_profit'] = df_clean['net_revenue'] * 0.6 - df_clean['marketing_spend']
    df_clean['profit_margin'] = (df_clean['estimated_profit'] / df_clean['net_revenue']) * 100

    # Identify outliers using IQR method
    def identify_outliers(series,factor=1.5) :
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return (series < lower_bound) | (series > upper_bound)

    # Mark outliers
    df_clean['quantity_outlier'] = identify_outliers(df_clean['quantity'])
    df_clean['revenue_outlier'] = identify_outliers(df_clean['net_revenue'])

    outlier_count = df_clean['quantity_outlier'].sum()
    print(f"Identified {outlier_count} quantity outliers ({outlier_count / len(df_clean) * 100:.1f}%)")

    # Data validation
    print("\nData validation checks:")

    # Check for negative values where they shouldn't exist
    negative_quantity = (df_clean['quantity'] < 0).sum()
    negative_price = (df_clean['unit_price'] < 0).sum()
    negative_revenue = (df_clean['net_revenue'] < 0).sum()

    print(f"  Negative quantities: {negative_quantity}")
    print(f"  Negative prices: {negative_price}")
    print(f"  Negative revenues: {negative_revenue}")

    # Check satisfaction score bounds
    invalid_satisfaction = ((df_clean['customer_satisfaction_score'] < 1) |
                            (df_clean['customer_satisfaction_score'] > 5)).sum()
    print(f"  Invalid satisfaction scores (not 1-5): {invalid_satisfaction}")

    # Summary of data preparation
    print(f"\n✓ Data cleaning completed")
    print(f"✓ Added {len([col for col in df_clean.columns if col not in dataframe.columns])} derived columns")
    print(f"✓ Final dataset shape: {df_clean.shape}")
    print()

    return df_clean


# Clean and prepare the data
df_clean = clean_and_prepare_data(df)


# =============================================================================
# STEP 6: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def perform_eda(dataframe) :
    """
    Perform comprehensive exploratory data analysis.
    """

    print("\nSTEP 6: EXPLORATORY DATA ANALYSIS")
    print("-" * 40)

    df = dataframe.copy()

    # Basic business metrics
    total_revenue = df['net_revenue'].sum()
    total_orders = len(df)
    avg_order_value = total_revenue / total_orders
    unique_customers = df['sale_id'].nunique()  # Using sale_id as proxy

    print("BUSINESS METRICS OVERVIEW:")
    print(f"  Total Revenue: ${total_revenue:,.2f}")
    print(f"  Total Orders: {total_orders:,}")
    print(f"  Average Order Value: ${avg_order_value:.2f}")
    print(f"  Date Range: {df['sale_date'].min().strftime('%Y-%m-%d')} to {df['sale_date'].max().strftime('%Y-%m-%d')}")

    # Product analysis
    print(f"\nPRODUCT PERFORMANCE:")
    product_stats = df.groupby('product_name').agg({
        'net_revenue' : ['sum','mean','count'],
        'quantity' : 'sum',
        'customer_satisfaction_score' : 'mean'
    }).round(2)

    product_stats.columns = ['total_revenue','avg_revenue','order_count','total_quantity','avg_satisfaction']
    product_stats = product_stats.sort_values('total_revenue',ascending=False)

    print(product_stats.head())

    # Regional analysis
    print(f"\nREGIONAL PERFORMANCE:")
    regional_stats = df.groupby('region_name').agg({
        'net_revenue' : ['sum','mean'],
        'customer_satisfaction_score' : 'mean',
        'sale_id' : 'count'
    }).round(2)

    regional_stats.columns = ['total_revenue','avg_revenue','avg_satisfaction','order_count']
    regional_stats = regional_stats.sort_values('total_revenue',ascending=False)

    print(regional_stats)

    # Channel analysis
    print(f"\nCHANNEL PERFORMANCE:")
    channel_stats = df.groupby('channel').agg({
        'net_revenue' : 'sum',
        'profit_margin' : 'mean',
        'customer_satisfaction_score' : 'mean'
    }).round(2)

    channel_stats = channel_stats.sort_values('net_revenue',ascending=False)
    print(channel_stats)

    # Time-based analysis
    print(f"\nTIME-BASED TRENDS:")
    monthly_trends = df.groupby(['year','month']).agg({
        'net_revenue' : 'sum',
        'sale_id' : 'count',
        'customer_satisfaction_score' : 'mean'
    }).round(2)

    monthly_trends.columns = ['revenue','orders','satisfaction']
    print("Recent monthly trends:")
    print(monthly_trends.tail())

    print()
    return product_stats,regional_stats,channel_stats,monthly_trends


# Perform EDA
product_stats,regional_stats,channel_stats,monthly_trends = perform_eda(df_clean)


# =============================================================================
# STEP 7: STATISTICAL ANALYSIS
# =============================================================================

def perform_statistical_analysis(dataframe) :
    """
    Perform statistical tests and advanced analytics.
    """

    print("\nSTEP 7: STATISTICAL ANALYSIS")
    print("-" * 40)

    df = dataframe.copy()

    # 1. Correlation Analysis
    print("1. CORRELATION ANALYSIS:")
    numeric_cols = ['quantity','unit_price','customer_age','customer_satisfaction_score',
                    'marketing_spend','net_revenue','profit_margin']
    correlation_matrix = df[numeric_cols].corr()

    # Find strongest correlations
    correlations = []
    for i,col1 in enumerate(numeric_cols) :
        for j,col2 in enumerate(numeric_cols[i + 1 :],i + 1) :
            corr_value = correlation_matrix.loc[col1,col2]
            correlations.append((col1,col2,corr_value))

    # Sort by absolute correlation value
    correlations.sort(key=lambda x : abs(x[2]),reverse=True)

    print("   Strongest correlations:")
    for col1,col2,corr in correlations[:5] :
        print(f"   {col1} ↔ {col2}: {corr:.3f}")

    # 2. Statistical Tests
    print(f"\n2. STATISTICAL TESTS:")

    # Test if satisfaction differs significantly across regions
    region_groups = [group['customer_satisfaction_score'].dropna().values
                     for name,group in df.groupby('region_name')]

    f_stat,p_value = stats.f_oneway(*region_groups)
    print(f"   ANOVA Test (Satisfaction vs Region):")
    print(f"   F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")

    if p_value < 0.05 :
        print("   ✓ Significant difference in satisfaction across regions")
    else :
        print("   ✗ No significant difference in satisfaction across regions")

    # Test if revenue differs across channels
    channel_groups = [group['net_revenue'].values for name,group in df.groupby('channel')]
    f_stat_channel,p_value_channel = stats.f_oneway(*channel_groups)
    print(f"   ANOVA Test (Revenue vs Channel):")
    print(f"   F-statistic: {f_stat_channel:.4f}, p-value: {p_value_channel:.4f}")

    # 3. Customer Segmentation using K-means
    print(f"\n3. CUSTOMER SEGMENTATION:")

    # Prepare features for clustering
    features_for_clustering = ['customer_age','net_revenue','customer_satisfaction_score','quantity']
    clustering_data = df[features_for_clustering].dropna()

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clustering_data)

    # Find optimal number of clusters using silhouette score
    silhouette_scores = []
    K_range = range(2,8)

    for k in K_range :
        kmeans = KMeans(n_clusters=k,random_state=42,n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        silhouette_avg = silhouette_score(scaled_features,cluster_labels)
        silhouette_scores.append(silhouette_avg)

    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"   Optimal number of clusters: {optimal_k}")
    print(f"   Silhouette score: {max(silhouette_scores):.3f}")

    # Perform final clustering
    kmeans_final = KMeans(n_clusters=optimal_k,random_state=42,n_init=10)
    df.loc[clustering_data.index,'customer_segment'] = kmeans_final.fit_predict(scaled_features)

    # Analyze segments
    segment_analysis = df.groupby('customer_segment').agg({
        'customer_age' : 'mean',
        'net_revenue' : 'mean',
        'customer_satisfaction_score' : 'mean',
        'quantity' : 'mean',
        'sale_id' : 'count'
    }).round(2)

    segment_analysis.columns = ['avg_age','avg_revenue','avg_satisfaction','avg_quantity','count']
    print(f"   Customer segments:")
    print(segment_analysis)

    print()
    return correlation_matrix,segment_analysis


# Perform statistical analysis
correlation_matrix,segment_analysis = perform_statistical_analysis(df_clean)


# =============================================================================
# STEP 8: DATA VISUALIZATION (CORRECTED VERSION)
# =============================================================================

def create_visualizations(dataframe,product_stats,regional_stats,correlation_matrix) :
    """
    Create comprehensive visualizations.

    FIXED: Improved layout, error handling, and chart readability
    """
    print("\nSTEP 8: CREATING VISUALIZATIONS")
    print("-" * 40)

    df = dataframe.copy()

    # FIXED: Use a more compatible matplotlib style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create a comprehensive dashboard with better spacing
    fig = plt.figure(figsize=(20,24))  # Increased figure size for better readability

    # Adjust subplot spacing
    plt.subplots_adjust(hspace=0.5,wspace=0.4)

    # 1. Product Revenue Analysis
    plt.subplot(4,3,1)
    product_revenue = product_stats['total_revenue'].head(6)
    bars = plt.bar(range(len(product_revenue)),product_revenue.values,
                   color='skyblue',alpha=0.8,edgecolor='navy',linewidth=1)
    plt.title('Revenue by Product',fontsize=14,fontweight='bold',pad=20)
    plt.xlabel('Product',fontsize=12)
    plt.ylabel('Revenue ($)',fontsize=12)
    plt.xticks(range(len(product_revenue)),
               [prod[:8] + '...' if len(prod) > 8 else prod for prod in product_revenue.index],
               rotation=45,ha='right')

    # Add value labels on bars
    for bar,value in zip(bars,product_revenue.values) :
        plt.text(bar.get_x() + bar.get_width() / 2,bar.get_height() + value * 0.02,
                 f'${value / 1000:.0f}K',ha='center',va='bottom',fontsize=10,fontweight='bold')
    plt.grid(axis='y',alpha=0.3)

    # 2. Regional Performance
    plt.subplot(4,3,2)
    regional_revenue = regional_stats['total_revenue']
    colors = plt.cm.Set3(np.linspace(0,1,len(regional_revenue)))
    wedges,texts,autotexts = plt.pie(regional_revenue.values,
                                     labels=[region.replace(' ','\n') if len(region) > 10 else region
                                             for region in regional_revenue.index],
                                     autopct='%1.1f%%',startangle=90,colors=colors,
                                     textprops={'fontsize' : 10})
    plt.title('Revenue Distribution by Region',fontsize=14,fontweight='bold',pad=20)

    # 3. Monthly Trends
    plt.subplot(4,3,3)
    # FIXED: Better monthly trend calculation
    df['year_month'] = df['sale_date'].dt.to_period('M')
    monthly_data = df.groupby('year_month')['net_revenue'].sum().sort_index()

    plt.plot(range(len(monthly_data)),monthly_data.values,
             marker='o',linewidth=3,markersize=8,color='green',markerfacecolor='lightgreen')
    plt.title('Monthly Revenue Trends',fontsize=14,fontweight='bold',pad=20)
    plt.xlabel('Month',fontsize=12)
    plt.ylabel('Revenue ($)',fontsize=12)

    # Better x-axis labels
    step = max(1,len(monthly_data) // 6)
    plt.xticks(range(0,len(monthly_data),step),
               [str(monthly_data.index[i]) for i in range(0,len(monthly_data),step)],
               rotation=45)
    plt.grid(True,alpha=0.3)

    # Format y-axis to show values in K
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,p : f'${x / 1000:.0f}K'))

    # 4. Customer Age Distribution
    plt.subplot(4,3,4)
    n,bins,patches = plt.hist(df['customer_age'],bins=20,color='lightgreen',
                              alpha=0.7,edgecolor='darkgreen',linewidth=1)
    plt.title('Customer Age Distribution',fontsize=14,fontweight='bold',pad=20)
    plt.xlabel('Age',fontsize=12)
    plt.ylabel('Frequency',fontsize=12)

    mean_age = df['customer_age'].mean()
    plt.axvline(mean_age,color='red',linestyle='--',linewidth=2,
                label=f'Mean: {mean_age:.1f}')
    plt.axvline(df['customer_age'].median(),color='orange',linestyle='--',linewidth=2,
                label=f'Median: {df["customer_age"].median():.1f}')
    plt.legend()
    plt.grid(axis='y',alpha=0.3)

    # 5. Satisfaction vs Revenue Scatter
    plt.subplot(4,3,5)
    # FIXED: Handle missing values in scatter plot
    scatter_data = df[['customer_satisfaction_score','net_revenue']].dropna()

    # Create scatter plot with color mapping based on revenue
    scatter = plt.scatter(scatter_data['customer_satisfaction_score'],
                          scatter_data['net_revenue'],
                          alpha=0.6,s=50,c=scatter_data['net_revenue'],
                          cmap='viridis',edgecolors='black',linewidth=0.5)

    plt.title('Satisfaction vs Revenue',fontsize=14,fontweight='bold',pad=20)
    plt.xlabel('Customer Satisfaction Score',fontsize=12)
    plt.ylabel('Net Revenue ($)',fontsize=12)
    plt.colorbar(scatter,label='Revenue ($)')

    # Add trend line if we have enough data
    if len(scatter_data) > 1 :
        z = np.polyfit(scatter_data['customer_satisfaction_score'],scatter_data['net_revenue'],1)
        p = np.poly1d(z)
        x_trend = np.linspace(scatter_data['customer_satisfaction_score'].min(),
                              scatter_data['customer_satisfaction_score'].max(),100)
        plt.plot(x_trend,p(x_trend),"r--",alpha=0.8,linewidth=3,
                 label=f'Trend (R²={np.corrcoef(scatter_data["customer_satisfaction_score"],scatter_data["net_revenue"])[0,1] ** 2:.3f})')
        plt.legend()
    plt.grid(True,alpha=0.3)

    # 6. Channel Performance
    plt.subplot(4,3,6)
    channel_revenue = df.groupby('channel')['net_revenue'].sum().sort_values(ascending=True)
    bars = plt.barh(range(len(channel_revenue)),channel_revenue.values,
                    color='coral',alpha=0.8,edgecolor='darkred',linewidth=1)
    plt.title('Revenue by Sales Channel',fontsize=14,fontweight='bold',pad=20)
    plt.xlabel('Revenue ($)',fontsize=12)
    plt.yticks(range(len(channel_revenue)),channel_revenue.index)

    # Add value labels
    for i,(bar,value) in enumerate(zip(bars,channel_revenue.values)) :
        plt.text(value + value * 0.02,bar.get_y() + bar.get_height() / 2,
                 f'${value / 1000:.0f}K',va='center',fontsize=10,fontweight='bold')
    plt.grid(axis='x',alpha=0.3)

    # 7. Correlation Heatmap
    plt.subplot(4,3,7)
    # FIXED: Handle correlation matrix properly
    if correlation_matrix is not None and not correlation_matrix.empty :
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix,dtype=bool))

        # Create heatmap
        sns.heatmap(correlation_matrix,mask=mask,annot=True,cmap='RdYlBu_r',center=0,
                    square=True,linewidths=0.5,cbar_kws={"shrink" : .8},fmt='.2f',
                    annot_kws={'fontsize' : 9})
        plt.title('Correlation Matrix',fontsize=14,fontweight='bold',pad=20)
        plt.xticks(rotation=45,ha='right')
        plt.yticks(rotation=0)
    else :
        plt.text(0.5,0.5,'Correlation Matrix\nNot Available',
                 ha='center',va='center',fontsize=12,transform=plt.gca().transAxes)
        plt.title('Correlation Matrix',fontsize=14,fontweight='bold',pad=20)

    # 8. Weekend vs Weekday Sales
    plt.subplot(4,3,8)
    weekend_sales = df.groupby('is_weekend')['net_revenue'].sum()
    labels = ['Weekday','Weekend']
    colors = ['lightblue','orange']

    bars = plt.bar(labels,weekend_sales.values,color=colors,alpha=0.8,
                   edgecolor='navy',linewidth=1)
    plt.title('Weekend vs Weekday Sales',fontsize=14,fontweight='bold',pad=20)
    plt.ylabel('Revenue ($)',fontsize=12)

    # Add percentage labels
    total_revenue = weekend_sales.sum()
    for bar,value in zip(bars,weekend_sales.values) :
        percentage = (value / total_revenue) * 100
        plt.text(bar.get_x() + bar.get_width() / 2,bar.get_height() + value * 0.02,
                 f'{percentage:.1f}%\n${value / 1000:.0f}K',
                 ha='center',va='bottom',fontsize=10,fontweight='bold')
    plt.grid(axis='y',alpha=0.3)

    # 9. Profit Margin Distribution
    plt.subplot(4,3,9)
    # FIXED: Handle infinite and missing values in profit margin
    profit_margins = df['profit_margin'].replace([np.inf,-np.inf],np.nan).dropna()

    if len(profit_margins) > 0 :
        n,bins,patches = plt.hist(profit_margins,bins=30,color='gold',alpha=0.7,
                                  edgecolor='darkorange',linewidth=1)
        plt.title('Profit Margin Distribution',fontsize=14,fontweight='bold',pad=20)
        plt.xlabel('Profit Margin (%)',fontsize=12)
        plt.ylabel('Frequency',fontsize=12)

        # Add vertical lines for mean and median
        mean_profit_margin = profit_margins.mean()
        median_profit_margin = profit_margins.median()
        plt.axvline(mean_profit_margin,color='red',linestyle='--',linewidth=2,
                    label=f'Mean: {mean_profit_margin:.1f}%')
        plt.axvline(median_profit_margin,color='blue',linestyle='--',linewidth=2,
                    label=f'Median: {median_profit_margin:.1f}%')
        plt.legend()
        plt.grid(axis='y',alpha=0.3)
    else :
        plt.text(0.5,0.5,'Profit Margin Data\nNot Available',
                 ha='center',va='center',fontsize=12,transform=plt.gca().transAxes)
        plt.title('Profit Margin Distribution',fontsize=14,fontweight='bold',pad=20)

    # 10. Quantity Distribution by Product
    plt.subplot(4,3,10)
    # Box plot showing quantity distribution by product
    product_quantity_data = [df[df['product_name'] == product]['quantity'].values
                             for product in df['product_name'].unique()]

    box_plot = plt.boxplot(product_quantity_data,labels=df['product_name'].unique(),
                           patch_artist=True)

    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0,1,len(box_plot['boxes'])))
    for patch,color in zip(box_plot['boxes'],colors) :
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.title('Quantity Distribution by Product',fontsize=14,fontweight='bold',pad=20)
    plt.xlabel('Product',fontsize=12)
    plt.ylabel('Quantity',fontsize=12)
    plt.xticks(rotation=45,ha='right')
    plt.grid(axis='y',alpha=0.3)

    # 11. Revenue vs Marketing Spend
    plt.subplot(4,3,11)
    plt.scatter(df['marketing_spend'],df['net_revenue'],alpha=0.6,s=50,
                c=df['customer_satisfaction_score'],cmap='RdYlGn',
                edgecolors='black',linewidth=0.5)
    plt.title('Revenue vs Marketing Spend',fontsize=14,fontweight='bold',pad=20)
    plt.xlabel('Marketing Spend ($)',fontsize=12)
    plt.ylabel('Net Revenue ($)',fontsize=12)

    # Add colorbar for satisfaction score
    cbar = plt.colorbar(label='Satisfaction Score')

    # Add trend line
    if len(df) > 1 :
        z = np.polyfit(df['marketing_spend'],df['net_revenue'],1)
        p = np.poly1d(z)
        x_trend = np.linspace(df['marketing_spend'].min(),df['marketing_spend'].max(),100)
        plt.plot(x_trend,p(x_trend),"r--",alpha=0.8,linewidth=2)
    plt.grid(True,alpha=0.3)

    # 12. Age Group Revenue Analysis
    plt.subplot(4,3,12)
    age_group_revenue = df.groupby('age_group')['net_revenue'].sum().sort_values(ascending=False)

    bars = plt.bar(range(len(age_group_revenue)),age_group_revenue.values,
                   color='mediumpurple',alpha=0.8,edgecolor='indigo',linewidth=1)
    plt.title('Revenue by Age Group',fontsize=14,fontweight='bold',pad=20)
    plt.xlabel('Age Group',fontsize=12)
    plt.ylabel('Revenue ($)',fontsize=12)
    plt.xticks(range(len(age_group_revenue)),age_group_revenue.index,rotation=45)

    # Add value labels
    for bar,value in zip(bars,age_group_revenue.values) :
        plt.text(bar.get_x() + bar.get_width() / 2,bar.get_height() + value * 0.02,
                 f'${value / 1000:.0f}K',ha='center',va='bottom',fontsize=10,fontweight='bold')
    plt.grid(axis='y',alpha=0.3)

    # Save the plot
    plt.tight_layout()
    plt.savefig('sales_analysis_dashboard.png',dpi=300,bbox_inches='tight',
                facecolor='white',edgecolor='none')

    print("✓ Dashboard saved as 'sales_analysis_dashboard.png'")
    plt.show()

    return fig


# Create visualizations (assuming the previous steps have been run)
try :
    visualization_fig = create_visualizations(df_clean,product_stats,regional_stats,correlation_matrix)
    print("✓ Visualizations created successfully!")
except Exception as e :
    print(f"✗ Error creating visualizations: {e}")
    print("Please ensure that previous steps have been executed and variables are defined.")


# =============================================================================
# STEP 9: ADVANCED ANALYTICS AND INSIGHTS
# =============================================================================

def generate_business_insights(dataframe,product_stats,regional_stats,channel_stats,segment_analysis=None) :
    """
    Generate actionable business insights from the analysis.
    """
    print("\nSTEP 9: GENERATING BUSINESS INSIGHTS")
    print("-" * 40)

    df = dataframe.copy()

    insights = []

    # Revenue insights
    total_revenue = df['net_revenue'].sum()
    top_product = product_stats['total_revenue'].index[0]
    top_product_revenue = product_stats['total_revenue'].iloc[0]
    top_product_share = (top_product_revenue / total_revenue) * 100

    insights.append(
        f"💰 TOP PRODUCT: {top_product} generates {top_product_share:.1f}% of total revenue (${top_product_revenue:,.2f})")

    # Regional insights
    top_region = regional_stats['total_revenue'].index[0]
    top_region_revenue = regional_stats['total_revenue'].iloc[0]
    top_region_share = (top_region_revenue / total_revenue) * 100

    insights.append(f"🌍 TOP REGION: {top_region} accounts for {top_region_share:.1f}% of total revenue")

    # Channel insights
    top_channel = channel_stats.index[0]
    top_channel_revenue = channel_stats['net_revenue'].iloc[0]
    top_channel_share = (top_channel_revenue / total_revenue) * 100

    insights.append(f"📱 TOP CHANNEL: {top_channel} drives {top_channel_share:.1f}% of total revenue")

    # Satisfaction insights
    avg_satisfaction = df['customer_satisfaction_score'].mean()
    high_satisfaction_revenue = df[df['customer_satisfaction_score'] >= 4]['net_revenue'].sum()
    high_satisfaction_share = (high_satisfaction_revenue / total_revenue) * 100

    insights.append(f"😊 SATISFACTION: Average score is {avg_satisfaction:.2f}/5.0")
    insights.append(
        f"⭐ HIGH SATISFACTION: Customers with 4+ ratings generate {high_satisfaction_share:.1f}% of revenue")

    # Age insights
    avg_age = df['customer_age'].mean()
    high_value_age_group = df.groupby('age_group')['net_revenue'].mean().idxmax()

    insights.append(f"👥 DEMOGRAPHICS: Average customer age is {avg_age:.1f} years")
    insights.append(f"💎 PREMIUM SEGMENT: {high_value_age_group} age group has highest average order value")

    # Seasonal insights
    weekend_revenue = df[df['is_weekend']]['net_revenue'].sum()
    weekend_share = (weekend_revenue / total_revenue) * 100

    insights.append(f"📅 TIMING: Weekend sales account for {weekend_share:.1f}% of total revenue")

    # Profit insights
    avg_profit_margin = df['profit_margin'].replace([np.inf,-np.inf],np.nan).mean()
    if not np.isnan(avg_profit_margin) :
        insights.append(f"📈 PROFITABILITY: Average profit margin is {avg_profit_margin:.1f}%")

    # Marketing efficiency
    marketing_roi = (total_revenue / df['marketing_spend'].sum()) if df['marketing_spend'].sum() > 0 else 0
    insights.append(f"📊 MARKETING ROI: ${marketing_roi:.2f} revenue per $1 marketing spend")

    # Print insights
    print("KEY BUSINESS INSIGHTS:")
    print("=" * 60)
    for i,insight in enumerate(insights,1) :
        print(f"{i:2d}. {insight}")

    print("\nRECOMMENDING:")
    print("=" * 60)

    # Generate recommendations based on insights
    recommendations = []

    if top_product_share > 40 :
        recommendations.append(
            f"🎯 DIVERSIFICATION: {top_product} dominates revenue. Consider diversifying product portfolio to reduce risk.")

    if avg_satisfaction < 4.0 :
        recommendations.append(
            f"🔧 SATISFACTION: Average satisfaction is {avg_satisfaction:.2f}. Focus on improving customer experience.")

    if weekend_share < 20 :
        recommendations.append("📅 WEEKEND BOOST: Weekend sales are low. Consider weekend-specific promotions.")

    if marketing_roi < 3 :
        recommendations.append("💡 MARKETING: Marketing ROI is low. Optimize marketing spend allocation.")

    low_performing_products = product_stats[product_stats['total_revenue'] < product_stats['total_revenue'].mean()]
    if len(low_performing_products) > 0 :
        recommendations.append(
            f"📉 UNDERPERFORMERS: {len(low_performing_products)} products are below average. Consider product optimization or discontinuation.")

    for i,rec in enumerate(recommendations,1) :
        print(f"{i:2d}. {rec}")

    if not recommendations :
        print("✅ Overall performance looks strong across all key metrics!")

    print()
    return insights,recommendations


# Generate business insights
try :
    business_insights,recommendations = generate_business_insights(
        df_clean,product_stats,regional_stats,channel_stats,segment_analysis
    )
except Exception as e :
    print(f"✗ Error generating insights: {e}")


# =============================================================================
# STEP 10: EXPORT RESULTS AND SUMMARY REPORT
# =============================================================================

def create_summary_report(dataframe,insights,recommendations,filename='sales_analysis_report.txt') :
    """
    Create a comprehensive summary report of the analysis.
    """
    print("\nSTEP 10: CREATING SUMMARY REPORT")
    print("-" * 40)

    df = dataframe.copy()

    try :
        with open(filename,'w',encoding='utf-8') as f :
            f.write("=" * 80 + "\n")
            f.write("SALES DATA ANALYSIS - COMPREHENSIVE REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"Analysis Period: {df['sale_date'].min().strftime('%Y-%m-%d')} to {df['sale_date'].max().strftime('%Y-%m-%d')}\n")
            f.write("\n")

            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Revenue: ${df['net_revenue'].sum():,.2f}\n")
            f.write(f"Total Orders: {len(df):,}\n")
            f.write(f"Average Order Value: ${df['net_revenue'].mean():.2f}\n")
            f.write(f"Unique Products: {df['product_name'].nunique()}\n")
            f.write(f"Geographic Coverage: {df['region_name'].nunique()} regions\n")
            f.write(f"Sales Channels: {df['channel'].nunique()}\n")
            f.write("\n")

            # Key Insights
            f.write("KEY BUSINESS INSIGHTS\n")
            f.write("-" * 40 + "\n")
            for i,insight in enumerate(insights,1) :
                # Remove emoji for text file
                clean_insight = ''.join(char for char in insight if ord(char) < 127)
                f.write(f"{i:2d}. {clean_insight}\n")
            f.write("\n")

            # Recommendations
            f.write("STRATEGIC RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            for i,rec in enumerate(recommendations,1) :
                # Remove emoji for text file
                clean_rec = ''.join(char for char in rec if ord(char) < 127)
                f.write(f"{i:2d}. {clean_rec}\n")
            f.write("\n")

            # Detailed Statistics
            f.write("DETAILED PERFORMANCE METRICS\n")
            f.write("-" * 40 + "\n")

            # Product Performance
            f.write("Product Performance:\n")
            for product,revenue in product_stats['total_revenue'].head().items() :
                f.write(f"  {product}: ${revenue:,.2f}\n")
            f.write("\n")

            # Regional Performance
            f.write("Regional Performance:\n")
            for region,revenue in regional_stats['total_revenue'].items() :
                f.write(f"  {region}: ${revenue:,.2f}\n")
            f.write("\n")

            # Channel Performance
            f.write("Channel Performance:\n")
            for channel,revenue in channel_stats['net_revenue'].items() :
                f.write(f"  {channel}: ${revenue:,.2f}\n")
            f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("End of Report\n")
            f.write("=" * 80 + "\n")

        print(f"✓ Summary report saved as '{filename}'")
        print(f"✓ Report contains {len(insights)} insights and {len(recommendations)} recommendations")

    except Exception as e :
        print(f"✗ Error creating summary report: {e}")


# Create summary report
try :
    create_summary_report(df_clean,business_insights,recommendations)
except Exception as e :
    print(f"✗ Error in report generation: {e}")

print("\n" + "=" * 80)
print("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("Files generated:")
print("  • sales_data.csv - Raw sales data")
print("  • sales_analysis_dashboard.png - Comprehensive visualizations")
print("  • sales_analysis_report.txt - Summary report with insights")
print("\nNext steps:")
print("  • Review the visualizations for patterns and trends")
print("  • Implement the strategic recommendations")
print("  • Set up automated reporting for ongoing analysis")
print("  • Consider A/B testing for optimization opportunities")
print("=" * 80)
