% 添加 SIFTflow 工具箱路径（请根据实际情况修改路径）
delete(gcp('nocreate'))
addpath(genpath('~/SIFTflow'));

% GT 图像文件夹，GT 图像文件名为 "01.png" ~ "50.png"
% 
% gtdir = 'E:\projects\cv\data\baselineDataSets\newBaseline\new_target\单栏';
%gtdir = 'E:\projects\cv\data\baselineDataSets\newBaseline\new_target\三栏以上复杂版面';
%gtdir = 'E:\projects\cv\data\baselineDataSets\newBaseline\new_target\国际票据';
%gtdir = 'E:\projects\cv\data\baselineDataSets\newBaseline\new_target\education';
gtdir = 'E:\projects\siggraph2025\data\baselineDataSets\newBaseline\new_target\book';


% 矫正图像文件夹，矫正图像文件名格式为 x_a_b_k_m_geo.png
%imdir = 'E:\projects\cv\data\baselineDataSets\evaluation_benchmark\dewarping_models\DocTr\evaluation\geo_rec';
%imdir = 'E:\projects\cv\data\baselineDataSets\evaluation_benchmark\dewarping_models\DocTr\evaluation\init_2\geo_rec';
%imdir = 'E:\projects\cv\data\baselineDataSets\evaluation_benchmark\dewarping_models\DocTr\evaluation\init_3';
%imdir = 'F:\evaluation_benchmark\dewarping_models\DocTr\evaluation\init_4\init_4';
%imdir = 'F:\evaluation_benchmark\dewarping_models\DocTr\evaluation\init_5';
%imdir = 'F:\evaluation_benchmark\dewarping_models\FTA\output_final\init_5';
%imdir = 'F:\evaluation_benchmark\dewarping_models\ddcp\output\init_5\2025-03-15\2025-03-15 01%3A53%3A23 @2021-02-03\144\test';
imdir = 'F:\evaluation_benchmark\dewarping_models\DewarpNet\output\init_5';
%imdir = 'F:\evaluation_benchmark\dewarping_models\PaperEdge\ls_output\init_5';
%imdir = "F:\evaluation_benchmark\dewarping_models\UVDOC\output_final\init_5";
%imdir = "F:\evaluation_benchmark\dewarping_models\ours_DVD\anyphoto\0707_5\dewarped_pred";
%imdir = "F:\evaluation_benchmark\dewarping_models\init_all_final\init_5";
%imdir = "F:\evaluation_benchmark\dewarping_models\DocGeoNet\save_path\init_5";


tarea = 598400;
res = cell(50, 1);  % 共 50 张 GT 图像

fixed_x = 5;  % 固定 x 值为 2

% 开启 8 个并行工作者
parpool(8);

parfor k = 1 : 100  % 遍历 100 张 GT 图像
    disp(k);
    % 读取 GT 图像，文件名格式为 01.png, 02.png, ... 50.png（使用两位格式读取）
    filename = fullfile(gtdir, sprintf('%02d.png', k));
    rimg = imread(filename);

    
    % 每张 GT 对应 3×2=6 张矫正图像，初始化存储矩阵 t（6行，5列）
    t = zeros(18, 5);
    idx = 1;  % 用于记录 t 的当前行号
    a=1;
    
    for b = 1 : 3   % b 从 1 到 3，共 3 组
        for m = 1 : 2   % 每组包含 2 张图像（m = 1 或 2）
            try
                % 构造矫正图像文件名，格式：x_a_b_k_m_geo.png
                % k 直接以普通整数输出（例如 "1", "2", ..., "100"）
                filename_corr = sprintf('%d_%d_%d_%d_%d.jpg', fixed_x, a, b, k, m);
                ximg = imread(fullfile(imdir, filename_corr));
                
                % 计算评估指标（调用外部函数 evalUnwarp 与 evalAlignedUnwarp）
                [ms, ld] = evalUnwarp(ximg, rimg);
                [~, relres] = evalAlignedUnwarp(ximg, rimg);
                
                % 将评估结果保存到 t 中： [GT编号, 序号, 对齐指标, MS指标, LD指标]
                t(idx, :) = [k, idx, relres, ms, ld];
            catch ME
                disp(ME.message);
                disp(['GT图像编号: ', num2str(k)]);
                % 若出现异常，则记录为 -1
                t(idx, :) = [k, idx, -1, -1, -1];
            end
            idx = idx + 1;
        end
    end
    res{k} = t;
end


% 将 cell 数组转换为矩阵
res = cell2mat(res);

% 筛选出有效的评估结果（第三列 > 0）
valres = res(res(:, 3) > 0, :);
% 计算有效数据各项指标的均值
avg = mean(valres, 1);
% 将均值行附加到结果矩阵的最后
res = cat(1, res, avg);

% 将结果保存为 ASCII 格式的文本文件，文件名为 adres.txt，存放在矫正图像文件夹下
save(fullfile(imdir, 'adres.txt'), 'res', '-ascii');
