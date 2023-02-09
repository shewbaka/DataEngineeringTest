### Notebook for Data Engineering Test


```python
## This is a helper script I have been running for a long time to provide as much width as needed (and is practical)
## Also provides simple mods for cell/output text size and several other hacks

%run display.py
get_html()
```


```python
# from collections import UserDict
# import types as types_from_types
# import pyspark.sql.functions as F
# from pyspark.sql import DataFrame, Row

from pyspark.sql.functions import *
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
```


```python
default_conf = {
    "spark.plugins": "org.apache.spark.sql.connect.SparkConnectPlugin"
}
overwrite_conf = {
    "spark.master": "master",
}
def create_conf(**kwargs: Any) -> SparkConf:
    sparkConf = SparkConf(**kwargs)
    for k, v in overwrite_conf.items():
        sparkConf.set(k, v)
    for k, v in default_conf.items():
        if not sparkConf.contains(k):
            sparkConf.set(k, v)
    return sparkConf
```


```python
## Fun for another time...

local_conf = {
    "spark.cores.max": 6,
    "spark.driver.cores": 2,
    "spark.driver.memory": "4g",
    "spark.executor.memory": "3g",
    "spark.executor.cores": 3,
    "spark.logConf": True,
    "spark.shuffle.service.enabled": True,
    "spark.dynamicAllocation.enabled": True,
    "spark.dynamicAllocation.minExecutors": 4,
    "spark.dynamicAllocation.maxExecutors": 6,
    "spark.default.parallelism": 16
}

conf = SparkConf(local_conf)
conf.setMaster("local[*]").setAppName("DataEngineeringTest")
sc = SparkContext(conf=conf)
spark = SparkSession(sparkContext=sc)
```


```python
col_doc_num                 = "DocumentNumber"
col_doc_dt                  = "DocumentDate"
col_doc_type                = "DocumentType"
col_ref_to_doc_num          = "RefersToDocumentNumber"
col_ref_to_doc_year         = "RefersToDocumentYear"
col_remarks                 = "Remarks"

col_formatted_doc_dt        = "FormattedDocumentDate"

col_ref_year_ref_num_arr    = "RefYearRefNumArray"
col_ref_year_num_concat     = "RefYearNumConcat"

col_year_num_arr            = "YearNumArray"
col_year_num_concat         = "YearNumConcat"

col_all                     = col('*')

key_types_jrt               = ('J','T','R')
key_types_abc               = ('A','B','C')
```


```python
def show_df(df, n=25, orderby_cols=(col_doc_type, col_ref_to_doc_num)):
    df.orderBy(*orderby_cols).show(n)
```


```python
df = spark.read.csv('data-engineer-interview-data.csv', header=True, inferSchema=True, samplingRatio=0.5)
```


```python
## "Make" the date from scratch using the existing repr, need to be careful of M/d/yyy vs. MM/dd/yyyy (e.g.)

df = df.select("*", split(col_doc_dt, "/").alias('dt')).withColumn(col_formatted_doc_dt,
        to_date(make_date(month=col("dt").__getitem__(0), day=col("dt").__getitem__(1), year=col("dt").__getitem__(2)), 'MM/dd/yyyy'))
df = df.orderBy(col_doc_type, col_doc_num).drop('dt').fillna('')
```


```python
## Since the two "Reference" cols had years and nums in both, I created this func() to ensure the year was first in the concatenation of "year.num" being handy later for
## the joins. I explored the data quite a bit and saw that the only length 4 entries between the two "Reference" cols were also within the min/max range of years that 
## substiantiated the "DocumentDate" field so, for this assignment I made the assumption that they had been entered in the opposing fields occasionally, as mentioned

def concat_cols(df, first_col, second_col, final_col_name):
    concat_ref_col = when(length(first_col) == 4,
                        concat_ws('.', first_col, second_col)).otherwise(
                            concat_ws('.', second_col, first_col))
    return df.withColumn(final_col_name, concat_ref_col)

df = concat_cols(df, col(col_ref_to_doc_num), col(col_ref_to_doc_year), col_ref_year_num_concat)
```


```python
df = concat_cols(df, year(col_formatted_doc_dt), col(col_doc_num), col_year_num_concat)
```


```python
type_keys = {row[0]: row[1] for row in df.groupBy(col(col_doc_type)).count().collect()}.keys()
others = sorted([t for t in type_keys if t not in key_types_jrt])

col_with_temp = lambda c: (c, f"{c}_TEMP")
only_temp_col = lambda c: col_with_temp(c)[-1]
temp_for_col = lambda c: (f"{c}_TEMP", c)
```


```python
def get_doctype_with_temp_col(df, doc_type_chars, cur_col_to_temp):
    return df.where(col(col_doc_type).isin(doc_type_chars)).withColumnRenamed(*col_with_temp(cur_col_to_temp))
```


```python
## Split off DataFrames per needs outlined in the docs, using helpers to make subsequent joins easier

dfj = get_doctype_with_temp_col(df, 'J', col_year_num_concat)
dft= get_doctype_with_temp_col(df, 'T', col_ref_year_num_concat)
dfabc = get_doctype_with_temp_col(df, others, col_year_num_concat)
dfr = get_doctype_with_temp_col(df, 'R', col_ref_year_num_concat)
dfj.count(), dft.count(), dfr.count(), dfabc.count()
```




    (20, 9, 28, 89)




```python
## Used anti joins, which is ideal for discounting rows from a larger spread, but it works fine this way also with union reconstruction later

dfj_applied = dfj.join(dft, on=[dfj[only_temp_col(col_year_num_concat)] == dft[only_temp_col(col_ref_year_num_concat)]], how="left_anti")
print(f"dfj count before anti_join: [{dfj.count()}], dfj_applied count after: [{dfj_applied.count()}]")
```

    dfj count before anti_join: [20], dfj_applied count after: [13]



```python
## These are the records that matched but due to the join type were not included in the result
## And there are indeed a total of [7] records

print("Row count: ", dfj.subtract(dfj_applied).count())
dfj.subtract(dfj_applied).orderBy(col_doc_num).show(n=5)
```

    Row count:  7
    +--------------+------------+------------+----------------------+--------------------+-------+---------------------+----------------+------------------+
    |DocumentNumber|DocumentDate|DocumentType|RefersToDocumentNumber|RefersToDocumentYear|Remarks|FormattedDocumentDate|RefYearNumConcat|YearNumConcat_TEMP|
    +--------------+------------+------------+----------------------+--------------------+-------+---------------------+----------------+------------------+
    |            37|   7/24/1987|           J|                  null|                null|       |           1987-07-24|                |           1987.37|
    |            52|  12/22/2006|           J|                  null|                null|       |           2006-12-22|                |           2006.52|
    |            73|  11/13/2003|           J|                  null|                null|       |           2003-11-13|                |           2003.73|
    |           125|  12/16/1993|           J|                  null|                null|       |           1993-12-16|                |          1993.125|
    |           133|  11/16/1995|           J|                  null|                null|       |           1995-11-16|                |          1995.133|
    +--------------+------------+------------+----------------------+--------------------+-------+---------------------+----------------+------------------+
    only showing top 5 rows
    



```python
dfabc_applied = dfabc.join(dfr, on=[dfabc[col_with_temp(col_year_num_concat)[-1]] == dfr[col_with_temp(col_ref_year_num_concat)[-1]]], how="left_anti")
print(f"dfabc count before anti_join: [{dfabc.count()}], dfabc_applied count after: [{dfabc_applied.count()}]")
```

    dfabc count before anti_join: [89], dfabc_applied count after: [63]



```python
## These are the records that matched but due to the join type were not included in the result
## And there are indeed a total of [26] records

print("Row count: ", dfabc.subtract(dfabc_applied).count())
dfabc.subtract(dfabc_applied).orderBy(col_doc_type, col_doc_num).show(n=5)
```

    Row count:  26
    +--------------+------------+------------+----------------------+--------------------+-----------+---------------------+----------------+------------------+
    |DocumentNumber|DocumentDate|DocumentType|RefersToDocumentNumber|RefersToDocumentYear|    Remarks|FormattedDocumentDate|RefYearNumConcat|YearNumConcat_TEMP|
    +--------------+------------+------------+----------------------+--------------------+-----------+---------------------+----------------+------------------+
    |             1|   10/2/2000|           A|                  null|                null|$10,000.00 |           2000-10-02|                |            2000.1|
    |             9|  12/19/1988|           A|                  null|                null|           |           1988-12-19|                |            1988.9|
    |            28|  10/14/1994|           A|                  null|                null|           |           1994-10-14|                |           1994.28|
    |            55|   5/19/1989|           A|                  null|                null|           |           1989-05-19|                |           1989.55|
    |            65|    7/2/1997|           A|                  null|                null|           |           1997-07-02|                |           1997.65|
    +--------------+------------+------------+----------------------+--------------------+-----------+---------------------+----------------+------------------+
    only showing top 5 rows
    



```python
## Could have done this with a comprehension but the result is still a Potato
df_final = (dfj_applied.withColumnRenamed(*temp_for_col(col_year_num_concat)).union(
    dfabc_applied.withColumnRenamed(*temp_for_col(col_year_num_concat)).union(
        dfr.withColumnRenamed(*temp_for_col(col_ref_year_num_concat)).union(
            dft.withColumnRenamed(*temp_for_col(col_ref_year_num_concat))
        ))))
```


```python
print(f"Row count of final Dataframe: {df_final.count()}, Row count of initial Dataframe: {df.count()}")
df.subtract(df_final).orderBy(col_doc_type, col_doc_num).show(5)
```

    Row count of final Dataframe: 113, Row count of initial Dataframe: 146
    +--------------+------------+------------+----------------------+--------------------+-----------+---------------------+----------------+-------------+
    |DocumentNumber|DocumentDate|DocumentType|RefersToDocumentNumber|RefersToDocumentYear|    Remarks|FormattedDocumentDate|RefYearNumConcat|YearNumConcat|
    +--------------+------------+------------+----------------------+--------------------+-----------+---------------------+----------------+-------------+
    |             1|   10/2/2000|           A|                  null|                null|$10,000.00 |           2000-10-02|                |       2000.1|
    |             9|  12/19/1988|           A|                  null|                null|           |           1988-12-19|                |       1988.9|
    |            28|  10/14/1994|           A|                  null|                null|           |           1994-10-14|                |      1994.28|
    |            55|   5/19/1989|           A|                  null|                null|           |           1989-05-19|                |      1989.55|
    |            65|    7/2/1997|           A|                  null|                null|           |           1997-07-02|                |      1997.65|
    +--------------+------------+------------+----------------------+--------------------+-----------+---------------------+----------------+-------------+
    only showing top 5 rows
    


More of a traditional audit trail can be integrated via Py4J to really get after the action happening in each jvm and to record each individual record that is or isn't included in the final result set. It seems kind of silly to set that up just for this. But, in production, we can get as granular as we want. We can debug and log/audit at the task level when needed. For now I've created some auditable data and you can clearly see how the join subtraction results add up to that of the final counts.
