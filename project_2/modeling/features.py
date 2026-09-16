
import logging 
from project_2.modeling.dataset import load_data
from project_2.modeling.train import split_data
from project_2.modeling.train import split_data
from project_2.modeling.train import transformers

def nominal_vars(df):
    logging.info("Dealing with the nominal variables")
    df['default'] = df['default'].map({'yes': 1, 'no': 0})
    df['housing'] = df['housing'].map({'yes': 1, 'no': 0})
    df['loan'] = df['loan'].map({'yes': 1, 'no': 0})
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    df["education"] = df["education"].replace("unknown", np.nan)
    df["job"] = df["job"].replace("unknown", np.nan)
    df["contact"] = df["contact"].replace("unknown", np.nan)
    df['job'] = df['job'].map({'management':0,'technician':1,'entrepreneur':2, 'blue-collar':3,
                'retired': 4, 'admin': 5,  'services': 6, 'self-employed': 7,
                'unemployed': 8,  'housemaid': 9, 'student': 10 })
    
    df['marital'] = df['marital'].map({'single':0,'married':1,'divorced':2})
    df['contact'] = df['contact'].map({'cellular': 0, 'telephone': 1})
    logging.info("Dealing with the nominal variables was successful")
    return df


def ordinal_vars(df):
    logging.info("Dealing with the ordinal variables")
    education_encoder= ce.OrdinalEncoder(cols=['education'],return_df=True,
                            mapping=[{'col':'education',
    'mapping':{'primary': 0,'secondary': 1,'tertiary': 2}}])
    
    month_encoder= ce.OrdinalEncoder(cols=['month'],return_df=True,
                            mapping=[{'col':'month',
    'mapping':{'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5,
            'jun': 6, 'jul': 7, 'aug': 8, 'oct': 10, 'nov': 11, 'dec': 12}}])
    df[['education']] = education_encoder.fit_transform(df[['education']]).astype('int')
        #[[]] used to indicate dataframe not series
    df[['month']] = month_encoder.fit_transform(df[['month']]).astype('int')
    logging.info("Dealing with the ordinal variables was successful")
    return df

def missing_data(df):
    missing = df.isna().sum()/len(df)
    matrix_missing = msno.matrix(df)
    plt.show()
    plt.savefig("../reports/figures/missing_matrix.png")
    logging.info("Run missingness successfully")
    return missing, matrix_missing

def histgrams(df):
    age = sns.histplot(df['age'])
    plt.savefig("../reports/figures/histogram_age.png")
    plt.show()
    balance = sns.histplot(df['balance'])
    plt.savefig("../reports/figures/histogram_balance.png")
    plt.show()
    duration = sns.histplot(df['duration'])
    plt.savefig("../reports/figures/histogram_duration.png")
    plt.show()
    day = sns.histplot(df['day'])
    plt.savefig("../reports/figures/histogram_day.png")
    logging.info("Histogram run successfully")
    plt.show()

    return age, balance, duration, day

def outlier_analysis(df, column):
    q1_q = df[column].quantile(0.25)
    q3_q = df[column].quantile(0.75)
    # Find the IQR
    IQR = q3_q - q1_q
    factor = 2.5 # 2.5 × IQR means you only mark really extreme points as outliers
    lower_limit_q = q1_q - IQR*factor
    upper_limit_q = q3_q + IQR*factor

    is_lower_q = df[column] < lower_limit_q

    is_higher_q = df[column] > upper_limit_q
    # Combine the masks to filter for outliers
    outliers = df[column][is_lower_q | is_higher_q] 
    # Count and print the number of outliers
    print(len(outliers))
    logging.info("Outliers run successfully")
    return outliers

def correlation_matrix(df):
    corr = df.select_dtypes(include='number').corr()
    plt.figure(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr))
    correlation_matrix = sns.heatmap(corr, center=0,
            mask=mask,
              linewidths=1,
              annot=True,
                fmt=".2f"
)
    plt.title("Correlation matrix")
    plt.savefig("../reports/figures/correlation_matrix.png")
    logging.info("Loading correlation successful")
    plt.show()
    return correlation_matrix


 if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df = load_data()
    nominal_vars(df)
    ordinal_vars(df)
    missing_data(df)
    histgrams(df)
    outlier_analysis(df,"age")
    outlier_analysis(df,"duration")
    outlier_analysis(df,"day")
    outlier_analysis(df,"balance")

    correlation_matrix(df)

    split_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    transformers(X_train, X_test)

    