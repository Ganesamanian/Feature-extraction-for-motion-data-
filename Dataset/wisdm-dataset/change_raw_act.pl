#! /usr/bin/perl


#perl change_raw_act.pl [phone/watch] [accel/gyro]

$device = $ARGV[0];
$sensor = $ARGV[1];

$SRC = "/Users/abbyoneill/Desktop/AR16/raw/$device/$sensor/";
$DEST = "/Users/abbyoneill/Desktop/AR16/formatted/$device/$sensor/";

#read all the files
opendir (DIR, $SRC);
@FI= readdir (DIR);
closedir (DIR);

foreach $f (@FI){
        #grab all of the txt files      
        if($f =~ m/.txt/){
                push (@files, $f);
        }
}

foreach $m (@files){

        #check file name and get uid and act label from each
        if($m =~ m/(\d+)_(\w+)_(\w+)_(\w+).txt/){
                $uid = $1;
                $sensor = $2;
                $act = $3;
                $device = $4;

                open (IN, "<".$SRC.$m);
                @line = <IN>;
                close IN;

                open(OUT, ">>".$DEST."data_".$uid."_".$sensor."_".$device.".txt");

                #remove line num and add data
                foreach $l (@line){
                        if($l =~ m/(\d+,\S+)(\s+)/){
                                $data = $1;

                                print OUT ($uid.",".$act.",".$data.";\n");
                        }
                }
                close (OUT);
        }
        @line = ();
}
