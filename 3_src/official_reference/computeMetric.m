function [dp,op,cle,auc,dp_20]= computeMetric(pd_boxes,ground_truth)
% % %%%%%%%%%%%%%%
% pd_boxes: resulted bouding boxes
% ground_truth: reference  bouding boxes
% dp: distance precision;
% op: overlapd precision;
% auc: area under curve of op;
% dp_20: distance precision @ 20 pixel;
% % %%%
distance_precision_threshold=1:50;
PASCAL_threshold=0.02:0.02:1;
op=zeros(1,length(PASCAL_threshold));
cle=zeros(1,length(PASCAL_threshold));
dp=zeros(1,length(distance_precision_threshold));
for j=1:length(distance_precision_threshold)
    [dp(j),op(j),cle(j)]= ...
        compute_performance_measures(pd_boxes, ground_truth,distance_precision_threshold(j),PASCAL_threshold(j));
end
auc=mean(op);
dp_20=dp(20);
end