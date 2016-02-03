package edu.stanford.cs246.kMeans;

import java.io.IOException;
import java.util.Arrays;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class kMeans extends Configured implements Tool {
   private final int NUM_DIM = 58;
   private static double[][] centroids;
   public static void main(String[] args) throws Exception {
      System.out.println(Arrays.toString(args));
      int res = ToolRunner.run(new Configuration(), new kMeans(), args);
      
      System.exit(res);
   }

   @Override
   public int run(String[] args) throws Exception {
      System.out.println(Arrays.toString(args));
      Job job = new Job(getConf(), "WordCount");
      job.setJarByClass(kMeans.class);

      job.setMapperClass(Map.class);
      job.setReducerClass(Reduce.class);

      job.setInputFormatClass(TextInputFormat.class);
      job.setOutputFormatClass(TextOutputFormat.class);
      
      job.setOutputKeyClass(IntWritable.class);
      job.setOutputValueClass(ArrayWritable.class);
 
      job.setMapOutputKeyClass(IntWritable.class);
      job.setMapOutputValueClass(ArrayWritable.class);
      
      FileInputFormat.addInputPath(job, new Path(args[0]));
      FileOutputFormat.setOutputPath(job, new Path(args[1]));

      job.waitForCompletion(true);
      
      return 0;
   }
   
   public static class Map extends Mapper<LongWritable, Text, IntWritable, ArrayWritable> {
      private final static IntWritable ONE = new IntWritable(1);
      private Text word = new Text();

      @Override
      public void map(LongWritable key, Text value, Context context)
              throws IOException, InterruptedException {
         String[] val_str_array = value.toString().split("\\s");
         DoubleWritable[] val_array = new DoubleWritable[NUM_DIM]
         for (int i = 0; i < NUM_DIM; i++) {
         	val_array[i].set(Double.parseDouble(val_str_array[i]));
         }
         double min_cost = Double.MAX_VALUE;
         int c = -1;
         for (int i = 0; i < centroids.length; i++) {
             double cost = euclideanDistance(centroids[i], val_array);
             if (cost < min_cost) {
             	min_cost = cost;
             	c = i;
             }
         }
         ArrayWritable outputArray = new ArrayWritable(DoubleWritable.class);
         context.write(c, outputArray);
      }
   }

   public static class Reduce extends Reducer<IntWritable, ArrayWritable, IntWritable, ArrayWritable> {
      @Override
      public void reduce(IntWritable key, Iterable<ArrayWritable> values, Context context)
              throws IOException, InterruptedException {
         int sum = 0;
         for (IntWritable val : values) {
            sum += val.get();
         }
         context.write(key, new IntWritable(sum));
      }
   }
   
}
