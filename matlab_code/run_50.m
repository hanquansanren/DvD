% 添加 SIFTflow 工具箱路径（请根据实际情况修改路径）
addpath(genpath('~/SIFTflow'));

% GT 图像文件夹，GT 图像文件名为 "01.png" ~ "50.png"
% 
%gtdir = 'E:\projects\cv\data\baselineDataSets\newBaseline\new_target\single_column';
%gtdir = 'E:\projects\cv\data\baselineDataSets\newBaseline\new_target\three_more_columns';
%gtdir = 'E:\projects\cv\data\baselineDataSets\newBaseline\new_target\commercial_invoice';
%gtdir = 'E:\projects\siggraph2025\data\baselineDataSets\newBaseline\new_target\education';
%gtdir = 'E:\projects\cv\data\baselineDataSets\newBaseline\new_target\book';
gtdir = 'E:\projects\siggraph2025\data\baselineDataSets\newBaseline\new_target\two_column';
%gtdir = 'E:\projects\cv\data\baselineDataSets\newBaseline\new_target\sparse_text';

% 矫正图像文件夹，矫正图像文件名格式为 x_a_b_k_m_geo.png
%imdir = 'E:\projects\cv\data\baselineDataSets\evaluation_benchmark\dewarping_models\DocTr\evaluation\geo_rec';
%imdir = 'E:\projects\cv\data\baselineDataSets\evaluation_benchmark\dewarping_models\DocTr\evaluation\init_2\geo_rec';
%imdir = 'E:\projects\cv\data\baselineDataSets\evaluation_benchmark\dewarping_models\DocTr\evaluation\init_3';
%imdir = 'F:\evaluation_benchmark\dewarping_models\DocTr\evaluation\init_4\init_4';
%imdir = 'F:\evaluation_benchmark\dewarping_models\DocTr\evaluation\init_5\init_5';
%imdir = 'F:\evaluation_benchmark\dewarping_models\DocTr\evaluation\init_6';
%imdir = 'F:\evaluation_benchmark\dewarping_models\DocTr\evaluation\init_7';

%imdir = 'F:\evaluation_benchmark\dewarping_models\FTA\save\init_1';
%imdir = 'F:\evaluation_benchmark\dewarping_models\FTA\save\init_2'
%imdir = 'F:\evaluation_benchmark\dewarping_models\FTA\save\init_3';
%imdir = 'F:\evaluation_benchmark\dewarping_models\FTA\save\init_4';
%imdir = 'F:\evaluation_benchmark\dewarping_models\FTA\save\init_5';
%imdir = 'F:\evaluation_benchmark\dewarping_models\FTA\save\init_6';
%imdir = 'F:\evaluation_benchmark\dewarping_models\FTA\save\init_7';

%imdir = 'F:\evaluation_benchmark\dewarping_models\ddcp\output\init_1\2025-03-09\2025-03-09 02%3A36%3A49 @2021-02-03\144\test';
%imdir = 'F:\evaluation_benchmark\dewarping_models\ddcp\output\init_2\2025-03-14\2025-03-14 03%3A02%3A11 @2021-02-03\144\test';
%imdir = 'F:\evaluation_benchmark\dewarping_models\ddcp\output\init_3\2025-03-14\2025-03-14 17%3A25%3A05 @2021-02-03\144\test';
%imdir = 'F:\evaluation_benchmark\dewarping_models\ddcp\output\init_4\2025-03-14\2025-03-14 18%3A37%3A32 @2021-02-03\144\test';
%imdir = 'F:\evaluation_benchmark\dewarping_models\ddcp\output\init_6\2025-03-15\2025-03-15 12%3A15%3A19 @2021-02-03\144\test';
%imdir = 'F:\evaluation_benchmark\dewarping_models\ddcp\output\init_7\2025-03-20\2025-03-20 15%3A03%3A13 @2021-02-03\144\test';

%imdir = 'F:\evaluation_benchmark\dewarping_models\DewarpNet\output\init_1';
%imdir = 'F:\evaluation_benchmark\dewarping_models\DewarpNet\output\init_2';
%imdir = 'F:\evaluation_benchmark\dewarping_models\DewarpNet\output\init_3';
%imdir = 'F:\evaluation_benchmark\dewarping_models\DewarpNet\output\init_4';
%imdir = 'F:\evaluation_benchmark\dewarping_models\DewarpNet\output\init_6';
%imdir = 'F:\evaluation_benchmark\dewarping_models\DewarpNet\output\init_7';

%imdir = 'F:\evaluation_benchmark\dewarping_models\PaperEdge\ls_output\init_1';
%imdir = 'F:\evaluation_benchmark\dewarping_models\PaperEdge\ls_output\init_3';
%imdir = 'F:\evaluation_benchmark\dewarping_models\PaperEdge\ls_output\init_4';  
%imdir = 'F:\evaluation_benchmark\dewarping_models\PaperEdge\ls_output\init_6';  
%imdir = 'F:\evaluation_benchmark\dewarping_models\PaperEdge\ls_output\init_7';

%imdir = 'F:\evaluation_benchmark\dewarping_models\UVDOC\save_path\init_1';
%imdir = 'F:\evaluation_benchmark\dewarping_models\UVDOC\save_path\init_3';
%imdir = 'F:\evaluation_benchmark\dewarping_models\UVDOC\save_path\init_4';


%imdir = "F:\evaluation_benchmark\dewarping_models\ours_DVD\anyphoto\0707_4\dewarped_pred";
imdir = "F:\evaluation_benchmark\dewarping_models\ours_DVD\anyphoto\0707_6\dewarped_pred";







tarea = 598400;
res = cell(50, 1);  % 共 50 张 GT 图像

fixed_x = 6;  % 固定 x 值为 2

% 开启 8 个并行工作者
parpool(8);

parfor k = 1 : 50  % 遍历 50 张 GT 图像
    disp(k);
    % 读取 GT 图像，文件名格式为 01.png, 02.png, ... 50.png（使用两位格式读取）
    filename = fullfile(gtdir, sprintf('%02d.png', k));
    rimg = imread(filename);

    
    % 每张 GT 对应 3×3×2 = 18 张矫正图像，初始化存储矩阵 t（18行，5列）
    t = zeros(18, 5);
    idx = 1;  % 用于记录 t 的当前行号
    
    for a = 1 : 3       % a 从 1 到 3
        for b = 1 : 3   % b 从 1 到 3，共 9 组
            for m = 1 : 2   % 每组包含 2 张图像（m = 1 或 2）
                try
                    % 构造矫正图像文件名，格式：x_a_b_k_m_geo.png
                    % k 直接以普通整数输出（例如 "1", "2", ..., "50"）
                    filename = sprintf('warped_%d_%d_%d_%d_%d.png', fixed_x, a, b, k, m);
                    ximg = imread(fullfile(imdir, filename));
                    
                    % 计算评估指标（调用外部函数 evalUnwarp 与 evalAlignedUnwarp）
                    [ms, ld] = evalUnwarp(ximg, rimg);
                    [~, relres] = evalAlignedUnwarp(ximg, rimg);
                    
                    % 将评估结果保存到 t 中： [GT编号, 序号, 对齐指标, MS指标, LD指标]
                    t(idx, :) = [k, idx, relres, ms, ld];
                catch ME
                    disp(ME.message);
                    disp(k);
                    % 若出现异常，则记录为 -1
                    t(idx, :) = [k, idx, -1, -1, -1];
                end
                idx = idx + 1;
            end
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
