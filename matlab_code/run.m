% 添加 SIFTflow 工具箱路径（请根据实际情况修改路径）
addpath(genpath('~/SIFTflow'));

% GT 图像文件夹，GT 图像文件名为 "01.png" ~ "50.png"
% 
% gtdir = 'E:\projects\cv\data\baselineDataSets\newBaseline\new_target\单栏';
%gtdir = 'E:\projects\cv\data\baselineDataSets\newBaseline\new_target\三栏以上复杂版面';
%gtdir = 'E:\projects\cv\data\baselineDataSets\newBaseline\new_target\国际票据';
%gtdir = 'E:\projects\cv\data\baselineDataSets\newBaseline\new_target\education';
%gtdir = 'E:\projects\cv\data\baselineDataSets\newBaseline\new_target\book';
%gtdir = 'E:\projects\cv\data\baselineDataSets\newBaseline\new_target\two_column'
%gtdir = 'E:\projects\cv\data\baselineDataSets\newBaseline\new_target\sparse_text';
gtdir  = 'E:\projects\siggraph2025\data\baselineDataSets\newBaseline\new_target\consumption_receipt';

% 矫正图像文件夹，矫正图像文件名格式为 x_a_b_k_m_geo.png
%imdir = 'E:\projects\cv\data\baselineDataSets\evaluation_benchmark\dewarping_models\DocTr\evaluation\geo_rec';
%imdir = 'E:\projects\cv\data\baselineDataSets\evaluation_benchmark\dewarping_models\DocTr\evaluation\init_2\geo_rec';
%imdir = 'E:\projects\cv\data\baselineDataSets\evaluation_benchmark\dewarping_models\DocTr\evaluation\init_3';
%imdir = 'F:\evaluation_benchmark\dewarping_models\DocTr\evaluation\init_4\init_4';
%imdir = 'F:\evaluation_benchmark\dewarping_models\DocTr\evaluation\init_5\init_5';
%imdir = 'F:\evaluation_benchmark\dewarping_models\DocTr\evaluation\init_6';
%imdir = 'F:\evaluation_benchmark\dewarping_models\DocTr\evaluation\init_7';
%imdir = 'F:\evaluation_benchmark\dewarping_models\DocTr\evaluation\init_8';
%imdir = 'F:\evaluation_benchmark\dewarping_models\FTA\save\init_8';
%imdir ='F:\evaluation_benchmark\dewarping_models\ddcp\output\init_8\2025-03-20\2025-03-20 15%3A41%3A55 @2021-02-03\144\test';
%imdir= 'F:\evaluation_benchmark\dewarping_models\DewarpNet\output\init_8';
%imdir = 'F:\evaluation_benchmark\dewarping_models\PaperEdge\ls_output\init_8';
imdir = "F:\evaluation_benchmark\dewarping_models\ours_DVD\anyphoto\0707_8\dewarped_pred";

tarea = 598400;


% 获取矫正图像文件夹中所有符合模式的文件
filePattern = fullfile(imdir, '*.png');
files = dir(filePattern);
nfiles = length(files);
fprintf('共找到 %d 个矫正图像文件\n', nfiles);


% 初始化结果矩阵，每行记录：[GT编号, idx, 对齐指标, MS指标, LD指标]
results = zeros(nfiles, 5);

% 开启 8 个并行工作者
parpool(8);

% 使用 parfor 遍历所有矫正图像文件
parfor i = 1:nfiles
    fileName = files(i).name;
    filePath = fullfile(imdir, fileName);
    
    % 使用正则表达式提取文件名中的数字
    % 文件名格式：x_a_b_k_m_geo.png
    tokens = regexp(fileName, 'warped_(\d+)_(\d+)_(\d+)_(\d+)_(\d+).png', 'tokens');
    if isempty(tokens)
        fprintf('文件名格式不匹配: %s\n', fileName);
        results(i,:) = [-1, -1, -1, -1, -1];
        continue;
    end
    tokens = tokens{1};  % 提取第一个匹配结果
    
    % 从文件名中提取参数，其中 k 为 GT 图像编号
    % 固定参数 fixed_x, a, b, m 在此不再记录
    fixed_x = str2double(tokens{1});
    a       = str2double(tokens{2});
    b       = str2double(tokens{3});
    k       = str2double(tokens{4});
    m       = str2double(tokens{5});
    
    % 构造对应的 GT 图像文件名（假设 GT 图像为两位数字，例如 "01.png"）
    gtFileName = fullfile(gtdir, sprintf('%02d.png', k));
    if exist(gtFileName, 'file') ~= 2
        fprintf('GT 图像不存在: %s\n', gtFileName);
        results(i,:) = [k, i, -1, -1, -1];
        continue;
    end
    rimg = imread(gtFileName);
    
    % 读取矫正图像
    try
        ximg = imread(filePath);
    catch ME
        fprintf('读取矫正图像出错 %s: %s\n', fileName, ME.message);
        results(i,:) = [k, i, -1, -1, -1];
        continue;
    end
    
    % 计算评估指标（调用外部函数 evalUnwarp 与 evalAlignedUnwarp）
    try
        [ms, ld] = evalUnwarp(ximg, rimg);
        [~, relres] = evalAlignedUnwarp(ximg, rimg);
    catch ME
        fprintf('计算评估指标出错 %s: %s\n', fileName, ME.message);
        results(i,:) = [k, i, -1, -1, -1];
        continue;
    end
    
    % 将本次计算结果保存到结果矩阵中
    % 记录：[GT编号, idx, 对齐指标, MS指标, LD指标]
    results(i,:) = [k, i, relres, ms, ld];
    
    fprintf('处理完成: %s\n', fileName);
end

% 可选：如果需要进一步筛选有效数据和计算均值，比如以对齐指标（第3列）> 0作为有效数据的判断条件
valres = results(results(:,3) > 0, :);
avg = mean(valres, 1);
results = cat(1, results, avg);

% 将结果保存为 ASCII 格式的文本文件，文件名为 results.txt，存放在矫正图像文件夹下
save(fullfile(imdir, 'adres.txt'), 'results', '-ascii');

fprintf('所有处理完成，结果保存在: %s\n', fullfile(imdir, 'results.txt'));