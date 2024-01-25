from pathlib import Path
# from timeeval import TimeEval, DatasetManager
import tensorflow as tf
from pyspark.sql import SparkSession, DataFrame, GroupedData
from pyspark.sql.streaming import StreamingQuery
from pyspark.sql.types import StructType
from pyspark.sql.functions import explode, split, window, to_timestamp, now, col, unix_timestamp
from datetime import datetime

def main() -> None:
    # 1) Set up DataStreamReader
    spark: SparkSession = SparkSession \
        .builder \
        .appName("StreamingAD") \
        .getOrCreate()
    userSchema: StructType = StructType() \
        .add("timestamp", "integer") \
        .add("value", "float") \
        .add("is_anomaly", "integer")
    dt_now = int(datetime.now().timestamp())
    data_stream: DataFrame = spark \
        .readStream \
        .option("sep", ",") \
        .schema(userSchema) \
        .csv("../data/test") \
        .select(col("timestamp").alias("id"), "value", "is_anomaly") \
        .withColumn("dt_timestamp", (unix_timestamp(to_timestamp("id")) + dt_now).cast('timestamp'))

    # 2) Set up Sliding window and uniform reservoir
    # def identity(pdf):
    #     return pdf
    window_duration, sw_slide_duration, maintain_state_for = 500, 1, 500
    sliding_window: DataFrame = data_stream \
        .withWatermark("dt_timestamp", f"{maintain_state_for} seconds") \
        .groupBy(
            window(data_stream.dt_timestamp,
                   f"5 minutes", f"1 seconds"),
            data_stream.value) \
        .count()
        # .applyInPandas(identity, schema="dt_timestamp timestamp, value float")

    # 3) Set up µ/sig - change, KS tests for SW, URES
    sw_query: StreamingQuery = sliding_window.writeStream \
        .outputMode("complete") \
        .format("console") \
        .start()
        # .format("parquet") \
        # .option("checkpointLocation", "out/checkpoints") \
        # .option("path", "out/sinks") \
    
    print(f'SW Query ID: {sw_query.id}')
    return sw_query
        

    # 4) Set up data representation query

    # 5) Connect models to data representation table

    # 6) Set up nonconformity + anomaly score and write to output folder

    # 7) Set up anomaly-aware reservoir

    # 8) Instantiate all µ/sig - change, KS tests for training sets \
    #    and call model retraining


if __name__ == '__main__':
    main()