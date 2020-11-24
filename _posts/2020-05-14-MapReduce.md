---
title: "MapReduce Poker Problem"
date: 2020-05-14
tags: [hadoop, mapreduce, aws]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Hadoop, Mapreduce, AWS"
mathjax: "true"
---

Develop and test a MapReduce-based approach in your Hadoop system to find all the missing Poker cards.

```java
package com.code.MapRed;

//IMPORTING JAVA PACKAGES AND HADOOP PACKAGES

import java.io.IOException;
import java.util.ArrayList;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

//MAIN CLASS

public class MapReduce extends Configured implements Tool{
    
    public int run(String[] args) throws Exception {
        Job job = Job.getInstance(getConf(), "MapReduce");
        job.setJarByClass(getClass());
        
        
        TextInputFormat.addInputPath(job, new Path(args[0]));
        job.setInputFormatClass(TextInputFormat.class);
        
        job.setMapperClass(MapClass.class);
        job.setReducerClass(ReduceClass.class);
        
        
        TextOutputFormat.setOutputPath(job, new Path(args[1]));
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        
        job.setOutputFormatClass(TextOutputFormat.class);
        
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        
        return job.waitForCompletion(true) ? 0 : 1;
    }
    
    public static void main(String[] args) throws Exception
    {
        int exitStatus = ToolRunner.run(new MapReduce(), args);
        System.exit(exitStatus);
    }
}

//MAPPER CLASS

class MapClass extends Mapper<LongWritable, Text, Text, IntWritable>
{
    Text _key = new Text();
    Text _value = new Text();
    
    public void map(LongWritable keys, Text values, Context context) throws IOException, InterruptedException
    {
        String _lines = values.toString();
        String[] _fields = _lines.split(" ");
        
        _key.set(_fields[0]);
        _value.set(_fields[1]);
        
        context.write(_key,new IntWritable(Integer.parseInt(_fields[1])));
    }
}

//REDUCER CLASS

class ReduceClass extends Reducer<Text, IntWritable, Text, Text>
{
    Text missingCardsText = new Text();
    
    public void reduce(Text token, Iterable<IntWritable> counts, Context context) throws IOException, InterruptedException
    {
        
        ArrayList<Integer> cardNumbers = new ArrayList<Integer>();
        
        int sum = 0;
        int temporaryValue = 0;
        
        for (IntWritable val : counts)
        {
            sum += val.get();
            temporaryValue = val.get();
            cardNumbers.add(temporaryValue);
        }
        
        StringBuilder _stringBuilderMissingCards = new StringBuilder();
        
        if(sum<91)
        {
            int i=1;
            while (i<=13){
                if(!cardNumbers.contains(i))
                    _stringBuilderMissingCards.append(i).append(",");
                i++;
            }
            missingCardsText.set(_stringBuilderMissingCards.substring(0,_stringBuilderMissingCards.length()-1));
        }
        else{
            missingCardsText.set("No missing card!");
        }
        context.write(token, missingCardsText);
    }
}
```
