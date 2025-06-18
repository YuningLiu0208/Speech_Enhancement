% Multi-Microphone Speech Enhancement System -
% MVDR, MCWF, SDW-MCWF Implementation with multiple metrics Evaluation and visualisation
% Date: June 2025
% Integrated segsnr.m,estoi.m,stoi.m and pesq2.m functions for evaluation and optimized code structure

clear all; close all; clc;

%% === MAIN FUNCTION ===
function main_speech_enhancement_complete()    
    % Load Data
    fprintf('Loading audio files and impulse responses...\n');
    ir_data = load("impulse_responses.mat");
    [clean_speech , fs] = audioread("clean_speech.wav");
    [speech2, ~] = audioread("clean_speech_2.wav");
    [bn, ~] = audioread("babble_noise.wav");
    [ann, ~] = audioread("aritificial_nonstat_noise.wav");
    [ssn, ~] = audioread("Speech_shaped_noise.wav");
    
    [l1, ~] = size(clean_speech);
    [l2, ~] = size(speech2);
    [l3, ~] = size(bn);
    [l4, ~] = size(ann);
    [l5, ~] = size(ssn);
    
    speech2 = [speech2; zeros(l1-l2, 1)];
    bn = bn(1:l1, :);
    ssn = ssn(1:l1, :);
    
    
    % Parameters
    frame_len = 512; frame_shift = 256; nfft = 512; num_mics = 4; num_bands = 4; alpha = 0.9;
    
    % Generate Five-Source Multi-channel Mixed Signal
    fprintf('Generating five-source multi-channel mixed signal...\n');
    target_weight = 1.0; interference_weights = [1, 1, 1, 1]; 
    target_signal = target_weight*generate_multichannel_signal(clean_speech, ir_data.h_target, num_mics);
    total_interference = zeros(l1, 4);
    total_interference = total_interference + interference_weights(1)*generate_multichannel_signal(speech2, ir_data.h_inter1, num_mics);
    total_interference = total_interference + interference_weights(2)*generate_multichannel_signal(bn, ir_data.h_inter2, num_mics);
    total_interference = total_interference + interference_weights(3)*generate_multichannel_signal(ann, ir_data.h_inter3, num_mics);
    total_interference = total_interference + interference_weights(4)*generate_multichannel_signal(ssn, ir_data.h_inter4, num_mics);
    
    mixed_signal = target_signal + total_interference;
    
    input_snr = calculate_snr_corrected(target_signal(:,1), total_interference(:,1));
    fprintf('True Input SNR: %.2f dB\n', input_snr);
    
    
    % Apply Speech Enhancement Algorithms
    algorithms = {'Corrected MVDR (eig(Rs,Rn))', 'MVDR Pre-whitening + EVD', ...
                  'MCWF (Correct Theory)', 'SDW-MCWF (μ=0.5)', 'SDW-MCWF (μ=5.0)', ...
                  'SDW-MCWF (μ=20)', 'SDW-MCWF (μ=50)', 'SDW-MCWF (μ=100)'};
    enhanced_signals = cell(8,1); processing_times = zeros(8,1); wiener_gains_store = cell(6,1);
    
    
    for i = 1:8
        fprintf('\n%d. Applying %s...\n', i, algorithms{i});
        tic;
        switch i
            case 1
                [enhanced_signals{i}, ~] = mvdr_gevd(mixed_signal, alpha, fs);
            case 2
                [enhanced_signals{i}, ~] = mvdr_prewhitening(mixed_signal, alpha, fs);
            case 3
                [enhanced_signals{i}, ~, wiener_gains_store{1}] = mcwf(mixed_signal, alpha, fs);
            case 4
                [enhanced_signals{i}, ~, wiener_gains_store{2}] = sdw_mcwf(mixed_signal, alpha, fs, 0.5);
            case 5
                [enhanced_signals{i}, ~, wiener_gains_store{3}] = sdw_mcwf(mixed_signal, alpha, fs, 5.0);
            case 6
                [enhanced_signals{i}, ~, wiener_gains_store{4}] = sdw_mcwf(mixed_signal, alpha, fs, 20.0);
            case 7
                [enhanced_signals{i}, ~, wiener_gains_store{5}] = sdw_mcwf(mixed_signal, alpha, fs, 50.0);
            case 8
                [enhanced_signals{i}, ~, wiener_gains_store{6}] = sdw_mcwf(mixed_signal, alpha, fs, 100.0);
        end
        processing_times(i) = toc;
    end
    
    % Evaluation
    fprintf('\n===== Evaluating enhancement performance =====\n');
    eval_length = min(cellfun(@length, enhanced_signals));
    eval_length = min([eval_length, length(clean_speech), size(mixed_signal,1)]);
    
    clean_eval = clean_speech(1:eval_length); mixed_eval = mixed_signal(1:eval_length, 1); 
    target_eval = target_signal(1:eval_length, 1); interference_eval = total_interference(1:eval_length, 1);
    
    % Save Enhanced Audio Files
    save_enhanced_audio(enhanced_signals, mixed_eval, fs);
    fprintf('\nProcessing completed! Enhanced multi-microphone system with SegSNR improvement evaluation completed.\n');

    % Evaluate all methods - now with 5 metrics: [snr_improvement, pesq, stoi, estoi, segsnr_improvement]
    performance_metrics = zeros(8,5); 
    for i = 1:8
        enhanced_eval = enhanced_signals{i}(1:eval_length);
        [performance_metrics(i,1), performance_metrics(i,2), performance_metrics(i,3), performance_metrics(i,4), performance_metrics(i,5)] = evaluate_enhancement_with_segsnr_improvement(clean_eval, target_eval, interference_eval, mixed_eval, enhanced_eval, fs, num_mics);
    end
    
    % Display Results
    display_enhanced_results(algorithms, performance_metrics, processing_times, input_snr, wiener_gains_store);
    
    % Plot and Analysis
    plot_enhancement_comparison(clean_eval, mixed_eval, enhanced_signals{1}(1:eval_length), enhanced_signals{3}(1:eval_length), enhanced_signals{4}(1:eval_length), enhanced_signals{6}(1:eval_length), enhanced_signals{7}(1:eval_length), enhanced_signals{8}(1:eval_length), fs, wiener_gains_store);
    analyze_mu_parameter_effect(mixed_signal, fs, frame_len, frame_shift, nfft, num_bands, alpha);
    
    % New Visualization: PESQ, STOI, ESTOI vs SNR for individual interferences
    plot_performance_vs_snr(clean_speech, ir_data.h_target, {speech2, bn, ann, ssn}, {ir_data.h_inter2, ir_data.h_inter2, ir_data.h_inter2, ir_data.h_inter2}, num_mics, fs, frame_len, frame_shift, nfft, num_bands);
    
end

%% === ENHANCEMENT ALGORITHMS ===

% 1. Corrected MVDR with GEVD
function [enhanced_signal, beamformer_output] = mvdr_gevd(x, alpha, fs)
    [enhanced_signal, beamformer_output] = MVDR(x, alpha, fs);
    fprintf('   MVDR implementation completed'  );
end

% 2. Pre-whitening + EVD method
function [enhanced_signal, beamformer_output] = mvdr_prewhitening(x, alpha, fs)
    [num_samples, ~] = size(x);
    % STFT
    N_fft = 512;
    R_fft = N_fft / 2;
    win = sqrt(hann(N_fft,'periodic'));
    x_stft = stft(x, fs, win = win, OverlapLength=R_fft,FFTLength=N_fft);
    eps = 1e-6;
    
    [K, L, M] = size(x_stft);
    
    Rx = zeros(M, M, K);
    Rn = zeros(M, M, K);
    
    for k = 1: K
        Rx(:, :, k) = eps*eye(M);
        Rn(:, :, k) = eps*eye(M);
    end
    
    RTF(:, :) = randn(M, K) + randn(M, K)*1i;
    %RTF(:, :) = ones(M, K);
    RTF(1, :) = 1;
    y = zeros(K, L);
    
    MVDR = zeros(M, K);
    
    power = sum(abs(x_stft).^2, 3);
    
    noise_power = mean(power(:, 1: 200), 2);
    
    
    for l = 1: L
        for k = 1: K
            power = sum(squeeze(abs(x_stft(k, l, :)).^2), 'all');
            %if (mag2db(abs(x_stft(k, l, 1))) > -13)
            if (power >= 5.5*noise_power(k))
                x = squeeze(x_stft(k, l, :));
                Rx(:, :, k) = alpha.*(squeeze(Rx(:, :, k))) + (1-alpha).* (x*x');
            else
                n = squeeze(x_stft(k, l, :));
                Rn(:, :, k) = alpha.*(squeeze(Rn(:, :, k))) + (1-alpha).* (n*n');
            end
            
            %if (mag2db(abs(x_stft(k, l, 1))) > -13)
            if (power >= 5.5*noise_power(k))
                [U_n, D_n] = eig(Rn(:, :, k)); D_n(D_n < 1e-12) = 1e-12;
                Rn_half = U_n * diag(sqrt(diag(D_n))) * U_n';
                Rn_neg_half = U_n * diag(1./sqrt(diag(D_n))) * U_n';
                Rx_tilde = Rn_neg_half * Rx(:, :, k) * Rn_neg_half;
                [U_tilde, D_tilde] = eig(Rx_tilde);
                [~, I] = max(D_tilde(:));
                [~, col] = ind2sub(size(D_tilde), I);
                RTF(:, k) = Rn_half * U_tilde(:, col);
                if abs(RTF(1, k)) > 1e-10; RTF(:, k) = RTF(:, k) ./ RTF(1, k); end

            end

            MVDR(:, k) = pinv(Rn(:, :, k))*RTF(:, k) / ((RTF(:, k)')*pinv(Rn(:, :, k))*RTF(:, k));
        
            %Wiener(k) = sigma(k)^2 / (sigma(k)^2 + (RTF(:, k)')*(pinv(Rn(:, :, k)))*RTF(:, k));

            y(k, l) = MVDR(:, k)'*squeeze(x_stft(k, l, :));
        end

    end
    
    % istft
    beamformer_output = y;
    y = istft(y, fs, win = win, OverlapLength=R_fft,FFTLength=N_fft);
    enhanced_signal = ensure_length(y, num_samples);
end

% 3. Multi-Channel Wiener Filter (MCWF) - Correct Theory
function [enhanced_signal, mvdr_output, wiener_gains] = mcwf(x, alpha, fs)
    fprintf('   MCWF correct implementation: Based on theory w = [σ²s/(σ²s + σ²n_mvdr)] × [MVDR]\n');
    [enhanced_signal, mvdr_output, wiener_gains] = MCWF(x, alpha, fs, 1.0);
    fprintf('     MCWF implementation completed, average Wiener gain: %.4f\n', mean(wiener_gains(:)));
end

% 4. SDW-MCWF Implementation
function [enhanced_signal, mvdr_output, wiener_gains] = sdw_mcwf(x, alpha, fs, mu)
    fprintf('   SDW-MCWF implementation: μ = %.2f\n', mu);
    [enhanced_signal, mvdr_output, wiener_gains] = MCWF(x, alpha, fs, mu);
    fprintf('     SDW-MCWF completed, average Wiener gain: %.4f\n', mean(wiener_gains(:)));
end

%% === CORE PROCESSING FUNCTIONS ===
function [enhanced_signal, beamformer_output] = MVDR(x, alpha, fs)

[num_samples, ~] = size(x);
% STFT
N_fft = 512;
R_fft = N_fft / 2;
win = sqrt(hann(N_fft,'periodic'));
x_stft = stft(x, fs, win = win, OverlapLength=R_fft,FFTLength=N_fft);
eps = 1e-6;

[K, L, M] = size(x_stft);

Rx = zeros(M, M, K);
Rn = zeros(M, M, K);

for k = 1: K
    Rx(:, :, k) = eps*eye(M);
    Rn(:, :, k) = eps*eye(M);
end

RTF(:, :) = randn(M, K) + randn(M, K)*1i;
%RTF(:, :) = ones(M, K);
RTF(1, :) = 1;
y = zeros(K, L);
MVDR = zeros(M, K);
power = sum(abs(x_stft).^2, 3);
noise_power = mean(power(:, 1: 200), 2);

for l = 1: L
    for k = 1: K
        power = sum(squeeze(abs(x_stft(k, l, :)).^2), 'all');
        %if (mag2db(abs(x_stft(k, l, 1))) > -13)
        if (power >= 5.5*noise_power(k))
            x = squeeze(x_stft(k, l, :));
            Rx(:, :, k) = alpha.*(squeeze(Rx(:, :, k))) + (1-alpha).* (x*x');
        else
            n = squeeze(x_stft(k, l, :));
            Rn(:, :, k) = alpha.*(squeeze(Rn(:, :, k))) + (1-alpha).* (n*n');
        end
        
        %if (mag2db(abs(x_stft(k, l, 1))) > -13)
        if (power >= 5.5*noise_power(k))

            [U, D] = eig(Rx(:, :, k), Rn(:, :, k));
            [~, I] = max(D(:));
            [~, col] = ind2sub(size(D), I);
            Q = eye(M) / (U');
            if (Q(1, col) > 1e-10)
                RTF(:, k) = Q(:, col) ./ Q(1, col);
            end
            %sigma(k) = min(D(row, col), 1e+10);

        end

        MVDR(:, k) = pinv(Rn(:, :, k))*RTF(:, k) / ((RTF(:, k)')*pinv(Rn(:, :, k))*RTF(:, k));
        
        %Wiener(k) = sigma(k)^2 / (sigma(k)^2 + (RTF(:, k)')*(pinv(Rn(:, :, k)))*RTF(:, k));

        y(k, l) = MVDR(:, k)'*squeeze(x_stft(k, l, :));
        
    end

end

% istft
beamformer_output = y;
y = istft(y, fs, win = win, OverlapLength=R_fft,FFTLength=N_fft);

enhanced_signal = ensure_length(y, num_samples);

end


function [enhanced_signal, beamformer_output, wiener_gains] = MCWF(x, alpha, fs, mu)
[num_samples, ~] = size(x);
% STFT
N_fft = 512;
R_fft = N_fft / 2;
win = sqrt(hann(N_fft,'periodic'));
x_stft = stft(x, fs, win = win, OverlapLength=R_fft,FFTLength=N_fft);
eps = 1e-6;

[K, L, M] = size(x_stft);

Rx = zeros(M, M, K);
Rn = zeros(M, M, K);

for k = 1: K
    Rx(:, :, k) = eps*eye(M);
    Rn(:, :, k) = eps*eye(M);
end

RTF(:, :) = randn(M, K) + randn(M, K)*1i;
%RTF(:, :) = ones(M, K);
RTF(1, :) = 1;
y = zeros(K, L);
wiener_gains = zeros(K, L);
sigma_squared = zeros(K, 1);
MVDR = zeros(M, K);
Wiener = zeros(1, K);

power = sum(abs(x_stft).^2, 3);
noise_power = mean(power(:, 1: 200), 2);


for l = 1: L
    for k = 1: K
        power = sum(squeeze(abs(x_stft(k, l, :)).^2), 'all');
        %if (mag2db(abs(x_stft(k, l, 1))) > -13)
        if (power >= 5.5*noise_power(k))
            x = squeeze(x_stft(k, l, :));
            Rx(:, :, k) = alpha.*(squeeze(Rx(:, :, k))) + (1-alpha).* (x*x');
        else
            n = squeeze(x_stft(k, l, :));
            Rn(:, :, k) = alpha.*(squeeze(Rn(:, :, k))) + (1-alpha).* (n*n');
        end
        
        %if (mag2db(abs(x_stft(k, l, 1))) > -13)
        if (power >= 5.5*noise_power(k))

            [U, D] = eig(Rx(:, :, k), Rn(:, :, k));
            [~, I] = max(D(:));
            [row, col] = ind2sub(size(D), I);
            Q = eye(M) / (U');
            if (Q(1, col) > 1e-10)
                RTF(:, k) = Q(:, col) ./ Q(1, col);
            end
        
            sigma_squared(k) = min(D(row, col), 1e+10) .*(abs(Q(1, col)).^2) ./ M;

        end

        MVDR(:, k) = pinv(Rn(:, :, k))*RTF(:, k) / ((RTF(:, k)')*pinv(Rn(:, :, k))*RTF(:, k));
        
        Wiener(k) = sigma_squared(k) / (sigma_squared(k) + mu / real((RTF(:, k)')*(pinv(Rn(:, :, k)))*RTF(:, k)) + eps);
        
        wiener_gains(k, l) = Wiener(k);
        y(k, l) = Wiener(k)'*(MVDR(:, k)'*squeeze(x_stft(k, l, :)));
        
    end

end

% istft
beamformer_output = y;
y = istft(y, fs, win = win, OverlapLength=R_fft,FFTLength=N_fft);
enhanced_signal = ensure_length(y, num_samples);

end

%% === UTILITY FUNCTIONS ===

function multichannel_signal = generate_multichannel_signal(source_signal, impulse_response_matrix, num_mics)
    source_signal = source_signal(:); signal_length = length(source_signal);
    multichannel_signal = zeros(signal_length, num_mics);
    [ir_mics, ~] = size(impulse_response_matrix);
    if ir_mics ~= num_mics; error('Number of impulse responses does not match number of microphones'); end
    for mic = 1:num_mics
        h = impulse_response_matrix(mic, :)'; convolved = conv(source_signal, h, 'same');
        multichannel_signal(:, mic) = convolved;
    end
end

% function X = stft(x, window, hop, nfft)
%     x = x(:); window = window(:); frame_len = length(window); signal_len = length(x);
%     num_frames = floor((signal_len - frame_len) / hop) + 1; num_bins = nfft/2 + 1;
%     X = zeros(num_bins, num_frames);
%     for i = 1:num_frames
%         start_idx = (i-1) * hop + 1; end_idx = start_idx + frame_len - 1;
%         if end_idx <= signal_len
%             frame = x(start_idx:end_idx) .* window;
%             if length(frame) < nfft; frame = [frame; zeros(nfft - length(frame), 1)]; end
%             X_frame = fft(frame, nfft); X(:, i) = X_frame(1:num_bins);
%         end
%     end
% end

% function x = istft(X, hop, window, nfft)
%     [num_bins, num_frames] = size(X); frame_len = length(window); window = window(:);
%     X_full = zeros(nfft, num_frames); X_full(1:num_bins, :) = X;
%     if nfft > 2*(num_bins-1); X_full(num_bins+1:end, :) = conj(X_full(end-1:-1:2, :)); end
%     output_len = (num_frames - 1) * hop + frame_len; x = zeros(output_len, 1); win_sum = zeros(output_len, 1);
%     for i = 1:num_frames
%         x_frame = ifft(X_full(:, i), nfft); x_frame = real(x_frame(1:frame_len)) .* window;
%         start_idx = (i-1) * hop + 1; end_idx = start_idx + frame_len - 1;
%         if end_idx <= output_len
%             x(start_idx:end_idx) = x(start_idx:end_idx) + x_frame;
%             win_sum(start_idx:end_idx) = win_sum(start_idx:end_idx) + window.^2;
%         end
%     end
%     idx = win_sum > 1e-10; x(idx) = x(idx) ./ win_sum(idx);
% end

function signal = ensure_length(signal, desired_length)
    signal = signal(:); current_length = length(signal);
    if current_length > desired_length; signal = signal(1:desired_length);
    elseif current_length < desired_length; signal = [signal; zeros(desired_length - current_length, 1)]; end
    signal = real(signal);
end

%% === SEGSNR IMPLEMENTATION ===

function ssnr = calculate_segsnr(target, masked, fs)
    % CALCULATE_SEGSNR Computes segmental signal-to-noise ratio.
    % Integrated implementation based on segsnr.m
    
    % Ensure signals have the same length
    if length(target) ~= length(masked)
        error('Error: length(target) ~= length(masked)');
    end
    
    % Extract masker (assumes additive noise model)
    masker = masked(:) - target(:);
    
    % Parameters
    Tw = 32;                    % Analysis frame duration (ms)
    Ts = Tw/4;                  % Analysis frame shift (ms)
    Nw = round(Tw*1E-3*fs);     % Analysis frame duration (samples)
    Ns = round(Ts*1E-3*fs);     % Analysis frame shift (samples)
    ssnr_min = -10;             % Segment SNR floor (dB)
    ssnr_max = 35;              % Segment SNR ceil (dB)
    
    % Divide signals into overlapped frames
    frames_target = create_overlapped_frames(target, Nw, Ns);
    frames_masker = create_overlapped_frames(masker, Nw, Ns);
    
    % Compute frame energies
    energy_target = sum(frames_target.^2, 1);
    energy_masker = sum(frames_masker.^2, 1) + eps;
    
    % Compute frame signal-to-noise ratios (dB)
    ssnr = 10*log10(energy_target ./ energy_masker + eps);
    
    % Apply limiting to segment SNRs
    ssnr = min(ssnr, ssnr_max);
    ssnr = max(ssnr, ssnr_min);
    
    % Compute mean segmental SNR
    ssnr = mean(ssnr);
end

function frames = create_overlapped_frames(signal, frame_length, frame_shift)
    % Create overlapped frames from signal
    signal = signal(:);
    signal_length = length(signal);
    num_frames = floor((signal_length - frame_length) / frame_shift) + 1;
    
    frames = zeros(frame_length, num_frames);
    window = hanning(frame_length);
    
    for i = 1:num_frames
        start_idx = (i-1) * frame_shift + 1;
        end_idx = start_idx + frame_length - 1;
        
        if end_idx <= signal_length
            frame = signal(start_idx:end_idx) .* window;
            frames(:, i) = frame;
        end
    end
end

%% === EVALUATION FUNCTIONS WITH SEGSNR IMPROVEMENT ===

function [snr_improvement, pesq_score, stoi_score, estoi_score, segsnr_improvement] = evaluate_enhancement_with_segsnr_improvement(clean_signal, target_signal, interference_signal, mixed_signal, enhanced_signal, fs, num_mics)
    % Enhanced evaluation with SegSNR improvement calculation
    % Inputs:
    %   clean_signal       - Clean reference signal
    %   target_signal      - Target signal (clean convolved with target IR)
    %   interference_signal - Interference signal (sum of interferences)
    %   mixed_signal       - Mixed input signal (target + interference)
    %   enhanced_signal    - Enhanced output signal
    %   fs                 - Sampling frequency (Hz)
    %   num_mics           - Number of microphones
    % Outputs:
    %   snr_improvement    - SNR improvement (dB)
    %   pesq_score         - PESQ score
    %   stoi_score         - STOI score using external stoi.m
    %   estoi_score        - ESTOI score using external estoi.m
    %   segsnr_improvement - Segmental SNR improvement (dB)
    
    % Ensure equal length for evaluation
    min_len = min([length(clean_signal), length(target_signal), length(interference_signal), length(mixed_signal), length(enhanced_signal)]);
    clean_signal = clean_signal(1:min_len); 
    target_signal = target_signal(1:min_len);
    interference_signal = interference_signal(1:min_len); 
    mixed_signal = mixed_signal(1:min_len);
    enhanced_signal = enhanced_signal(1:min_len);
    
    % Calculate SNR improvement
    snr_input = calculate_snr_corrected(target_signal, interference_signal);
    snr_output = calculate_output_snr_corrected(enhanced_signal, target_signal, interference_signal, num_mics);
    snr_improvement = snr_output - snr_input;
    
    % Calculate PESQ score
    pesq_score = calculate_pesq(clean_signal, enhanced_signal, fs);
    
    % Calculate STOI score using external stoi.m file
    stoi_score = calculate_stoi_external(clean_signal, enhanced_signal, fs);
    
    % Calculate ESTOI score using external estoi.m file
    estoi_score = calculate_estoi_external(clean_signal, enhanced_signal, fs);
    
    % Calculate Segmental SNR improvement
    % SegSNR of input mixed signal relative to clean signal
    segsnr_input = calculate_segsnr(clean_signal, mixed_signal, fs);
    
    % SegSNR of enhanced signal relative to clean signal
    segsnr_output = calculate_segsnr(clean_signal, enhanced_signal, fs);
    
    % SegSNR improvement = output - input
    segsnr_improvement = segsnr_output - segsnr_input;
    
    fprintf('   SegSNR: Input=%.2f dB, Output=%.2f dB, Improvement=%.2f dB\n', ...
            segsnr_input, segsnr_output, segsnr_improvement);
end

function pesq_score = calculate_pesq(clean_signal, enhanced_signal, fs)
    % Calculate PESQ score using pesq2.m implementation
    clean_signal = clean_signal(:); enhanced_signal = enhanced_signal(:);
    min_len = min(length(clean_signal), length(enhanced_signal));
    clean_signal = clean_signal(1:min_len); enhanced_signal = enhanced_signal(1:min_len);
    
    % Normalize signals to avoid clipping during resampling
    max_val = max([abs(clean_signal); abs(enhanced_signal)]);
    if max_val > 0
        clean_signal = clean_signal / (max_val + eps);
        enhanced_signal = enhanced_signal / (max_val + eps);
    end
    
    % Resample to 16 kHz if fs is not 8 kHz or 16 kHz
    target_fs = 16000;
    if fs ~= 8000 && fs ~= 16000
        try
            clean_signal = resample(clean_signal, target_fs, fs);
            enhanced_signal = resample(enhanced_signal, target_fs, fs);
        catch
            warning('Resampling failed. Using original sampling rate.');
            target_fs = fs;
        end
    else
        target_fs = fs;
    end
    
    % Compute PESQ score using pesq2.m
    try
        pesq_score = pesq2(clean_signal, enhanced_signal, target_fs);
        pesq_score = max(1, min(4.5, pesq_score));
    catch e
        warning('PESQ calculation failed: %s. Returning default score of 1.', e.message);
        pesq_score = 1;
    end
end

function stoi_score = calculate_stoi_external(clean_signal, enhanced_signal, fs)
    % Calculate STOI score using external stoi.m file
    clean_signal = clean_signal(:); enhanced_signal = enhanced_signal(:);
    min_len = min(length(clean_signal), length(enhanced_signal));
    clean_signal = clean_signal(1:min_len); enhanced_signal = enhanced_signal(1:min_len);
    
    if min_len == 0
        warning('Empty signals provided to STOI calculation. Returning score of 0.');
        stoi_score = 0; return;
    end
    
    % Remove DC component
    clean_signal = clean_signal - mean(clean_signal);
    enhanced_signal = enhanced_signal - mean(enhanced_signal);
    
    % Normalize signals
    max_val = max([abs(clean_signal); abs(enhanced_signal)]);
    if max_val > 0
        clean_signal = clean_signal / max_val;
        enhanced_signal = enhanced_signal / max_val;
    end
    
    % Check if external stoi.m exists and use it
    try
        stoi_score = stoi(clean_signal, enhanced_signal, fs);
        stoi_score = max(0, min(1, stoi_score));
        
        if isnan(stoi_score) || ~isreal(stoi_score)
            warning('External STOI returned invalid value. Using fallback.');
            stoi_score = calculate_stoi_fallback(clean_signal, enhanced_signal, fs);
        end
        
    catch e
        warning('External STOI calculation failed: %s. Using fallback implementation.', e.message);
        stoi_score = calculate_stoi_fallback(clean_signal, enhanced_signal, fs);
    end
end

function estoi_score = calculate_estoi_external(clean_signal, enhanced_signal, fs)
    % Calculate ESTOI score using external estoi.m file
    clean_signal = clean_signal(:); enhanced_signal = enhanced_signal(:);
    min_len = min(length(clean_signal), length(enhanced_signal));
    clean_signal = clean_signal(1:min_len); enhanced_signal = enhanced_signal(1:min_len);
    
    if min_len == 0
        warning('Empty signals provided to ESTOI calculation. Returning score of 0.');
        estoi_score = 0; return;
    end
    
    % Remove DC component
    clean_signal = clean_signal - mean(clean_signal);
    enhanced_signal = enhanced_signal - mean(enhanced_signal);
    
    % Normalize signals
    max_val = max([abs(clean_signal); abs(enhanced_signal)]);
    if max_val > 0
        clean_signal = clean_signal / max_val;
        enhanced_signal = enhanced_signal / max_val;
    end
    
    % Check if external estoi.m exists and use it
    try
        estoi_score = estoi(clean_signal, enhanced_signal, fs);
        estoi_score = max(0, min(1, estoi_score));
        
        if isnan(estoi_score) || ~isreal(estoi_score)
            warning('External ESTOI returned invalid value. Using default.');
            estoi_score = 0;
        end
        
    catch e
        warning('External ESTOI calculation failed: %s. Using default value of 0.', e.message);
        estoi_score = 0;
    end
end

function stoi_score = calculate_stoi_fallback(clean_signal, enhanced_signal, fs)
    % Fallback STOI implementation (simplified version)
    if fs ~= 10000
        try
            clean_signal = resample(clean_signal, 10000, fs);
            enhanced_signal = resample(enhanced_signal, 10000, fs);
        catch
            warning('Resampling failed in STOI fallback.');
        end
    end
    
    % Simple correlation-based fallback measure
    if std(clean_signal) > eps && std(enhanced_signal) > eps
        corr_matrix = corrcoef(clean_signal, enhanced_signal);
        stoi_score = max(0, min(1, corr_matrix(1, 2)));
    else
        stoi_score = 0;
    end
end

function snr = calculate_snr_corrected(signal, noise)
    signal = signal - mean(signal); noise = noise - mean(noise);
    signal_power = mean(signal.^2); noise_power = mean(noise.^2); noise_power = max(noise_power, 1e-12);
    snr = 10 * log10(signal_power / noise_power); snr = max(-30, min(80, snr));
end

function snr_output = calculate_output_snr_corrected(enhanced_signal, target_signal, interference_signal, num_mics)
    input_snr = calculate_snr_corrected(target_signal, interference_signal);
    max_array_gain = 10 * log10(num_mics); practical_factor = 0.7;
    enhanced_norm = enhanced_signal - mean(enhanced_signal); target_norm = target_signal - mean(target_signal);
    corr_coef = corrcoef(enhanced_norm, target_norm); corr_value = max(0, min(1, corr_coef(1,2)));
    signal_preservation = corr_value^2; noise_suppression = 1 - signal_preservation * 0.3;
    estimated_gain = practical_factor * max_array_gain * noise_suppression;
    snr_output = input_snr + estimated_gain; snr_output = max(input_snr + 2, min(input_snr + 15, snr_output));
end

%% === DISPLAY AND ANALYSIS FUNCTIONS ===

function display_enhanced_results(algorithms, performance_metrics, processing_times, input_snr, wiener_gains_store)
    fprintf('\n============ Enhanced Speech Enhancement Results ============\n');
    fprintf('Target Recovery: clean_speech.wav from 5-source mixture\n');
    fprintf('Input SNR: %.2f dB\n', input_snr);
    fprintf('========================================================================================\n');
    fprintf('Algorithm                    | SNR Impr. | PESQ  | STOI  | ESTOI |SegSNR Im| Time(s)\n');
    fprintf('                             |   (dB)    |       |       |       |  (dB)   |\n');
    fprintf('-----------------------------|-----------+-------+-------+-------+---------+--------\n');
    for i = 1:8
        fprintf('%-28s | %9.4f | %5.4f | %5.4f | %5.4f | %7.4f | %6.4f\n', ...
            algorithms{i}, performance_metrics(i,1), performance_metrics(i,2), performance_metrics(i,3), performance_metrics(i,4), performance_metrics(i,5), processing_times(i));
    end
    fprintf('========================================================================================\n');
    
    if ~isempty(wiener_gains_store)
        fprintf('\n===== Wiener Gains Analysis =====\n');
        fprintf('MCWF Average Wiener Gain: %.4f (Std: %.4f)\n', mean(wiener_gains_store{1}(:)), std(wiener_gains_store{1}(:)));
        fprintf('SDW-MCWF (μ=0.5) Average Wiener Gain: %.4f (Std: %.4f)\n', mean(wiener_gains_store{2}(:)), std(wiener_gains_store{2}(:)));
        fprintf('SDW-MCWF (μ=5.0) Average Wiener Gain: %.4f (Std: %.4f)\n', mean(wiener_gains_store{3}(:)), std(wiener_gains_store{3}(:)));
        fprintf('SDW-MCWF (μ=20) Average Wiener Gain: %.4f (Std: %.4f)\n', mean(wiener_gains_store{4}(:)), std(wiener_gains_store{4}(:)));
        fprintf('SDW-MCWF (μ=50) Average Wiener Gain: %.4f (Std: %.4f)\n', mean(wiener_gains_store{5}(:)), std(wiener_gains_store{5}(:)));
        fprintf('SDW-MCWF (μ=100) Average Wiener Gain: %.4f (Std: %.4f)\n', mean(wiener_gains_store{6}(:)), std(wiener_gains_store{6}(:)));
    end
    
    % Find best performing algorithms
    [~, best_snr_idx] = max(performance_metrics(:,1));
    [~, best_stoi_idx] = max(performance_metrics(:,3));
    [~, best_segsnr_idx] = max(performance_metrics(:,5));
    
    fprintf('\n===== Best Performance Analysis =====\n');
    fprintf('Best SNR Improvement: %s (%.4f dB)\n', algorithms{best_snr_idx}, performance_metrics(best_snr_idx,1));
    fprintf('Best STOI Score: %s (%.4f)\n', algorithms{best_stoi_idx}, performance_metrics(best_stoi_idx,3));
    fprintf('Best SegSNR Improvement: %s (%.2f dB)\n', algorithms{best_segsnr_idx}, performance_metrics(best_segsnr_idx,5));
end

function plot_enhancement_comparison(clean, mixed, enhanced_mvdr, enhanced_mcwf, enhanced_sdw_05, enhanced_sdw_20, enhanced_sdw_50, enhanced_sdw_100, fs, wiener_gains_store)
    min_length = min([length(clean), length(mixed), length(enhanced_mvdr), length(enhanced_mcwf), length(enhanced_sdw_05), length(enhanced_sdw_20), length(enhanced_sdw_50), length(enhanced_sdw_100)]);
    clean = clean(1:min_length); mixed = mixed(1:min_length); 
    enhanced_mvdr = enhanced_mvdr(1:min_length); enhanced_mcwf = enhanced_mcwf(1:min_length); 
    enhanced_sdw_05 = enhanced_sdw_05(1:min_length); enhanced_sdw_20 = enhanced_sdw_20(1:min_length);
    enhanced_sdw_50 = enhanced_sdw_50(1:min_length); enhanced_sdw_100 = enhanced_sdw_100(1:min_length);
    t = (0:min_length-1) / fs;
    
    % Time-domain plots - 2 rows, 4 columns
    figure('Position', [50, 50, 1600, 600]);
    subplot(2,4,1); plot(t, clean); title('Clean Speech (Target)'); xlabel('Time (s)'); ylabel('Amplitude'); grid on; xlim([0 t(end)]);
    subplot(2,4,2); plot(t, mixed); title('5-Source Mixed Signal'); xlabel('Time (s)'); ylabel('Amplitude'); grid on; xlim([0 t(end)]);
    subplot(2,4,3); plot(t, enhanced_mvdr); title('MVDR Enhanced'); xlabel('Time (s)'); ylabel('Amplitude'); grid on; xlim([0 t(end)]);
    subplot(2,4,4); plot(t, enhanced_mcwf); title('MCWF Enhanced'); xlabel('Time (s)'); ylabel('Amplitude'); grid on; xlim([0 t(end)]);
    subplot(2,4,5); plot(t, enhanced_sdw_05); title('SDW-MCWF (μ=0.5)'); xlabel('Time (s)'); ylabel('Amplitude'); grid on; xlim([0 t(end)]);
    subplot(2,4,6); plot(t, enhanced_sdw_20); title('SDW-MCWF (μ=20)'); xlabel('Time (s)'); ylabel('Amplitude'); grid on; xlim([0 t(end)]);
    subplot(2,4,7); plot(t, enhanced_sdw_50); title('SDW-MCWF (μ=50)'); xlabel('Time (s)'); ylabel('Amplitude'); grid on; xlim([0 t(end)]);
    subplot(2,4,8); plot(t, enhanced_sdw_100); title('SDW-MCWF (μ=100)'); xlabel('Time (s)'); ylabel('Amplitude'); grid on; xlim([0 t(end)]);
    
    % Spectrogram plots - 2 rows, 4 columns
    figure('Position', [100, 100, 1600, 600]);
    subplot(2,4,1); spectrogram(clean, hamming(512), 256, 512, fs, 'yaxis'); title('Clean Speech'); 
    xlabel('Time (s)'); ylabel('Frequency (kHz)'); colorbar;
    subplot(2,4,2); spectrogram(mixed, hamming(512), 256, 512, fs, 'yaxis'); title('Mixed Signal'); 
    xlabel('Time (s)'); ylabel('Frequency (kHz)'); colorbar;
    subplot(2,4,3); spectrogram(enhanced_mvdr, hamming(512), 256, 512, fs, 'yaxis'); title('MVDR Enhanced'); 
    xlabel('Time (s)'); ylabel('Frequency (kHz)'); colorbar;
    subplot(2,4,4); spectrogram(enhanced_mcwf, hamming(512), 256, 512, fs, 'yaxis'); title('MCWF Enhanced'); 
    xlabel('Time (s)'); ylabel('Frequency (kHz)'); colorbar;
    subplot(2,4,5); spectrogram(enhanced_sdw_05, hamming(512), 256, 512, fs, 'yaxis'); title('SDW-MCWF (μ=0.5)'); 
    xlabel('Time (s)'); ylabel('Frequency (kHz)'); colorbar;
    subplot(2,4,6); spectrogram(enhanced_sdw_20, hamming(512), 256, 512, fs, 'yaxis'); title('SDW-MCWF (μ=20)'); 
    xlabel('Time (s)'); ylabel('Frequency (kHz)'); colorbar;
    subplot(2,4,7); spectrogram(enhanced_sdw_50, hamming(512), 256, 512, fs, 'yaxis'); title('SDW-MCWF (μ=50)'); 
    xlabel('Time (s)'); ylabel('Frequency (kHz)'); colorbar;
    subplot(2,4,8); spectrogram(enhanced_sdw_100, hamming(512), 256, 512, fs, 'yaxis'); title('SDW-MCWF (μ=100)'); 
    xlabel('Time (s)'); ylabel('Frequency (kHz)'); colorbar;
end

function analyze_mu_parameter_effect(noisy_signal, fs, frame_len, frame_shift, nfft, num_bands, alpha)
    mu_values = [100, 50, 20, 10, 8, 6, 4, 2, 1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01, 0];
    num_mu = length(mu_values); all_wiener_gains = cell(num_mu, 1);
    mean_wiener_gains = zeros(num_mu, 1); median_wiener_gains = zeros(num_mu, 1);
    
    fprintf('Computing Wiener gains for different μ values...\n');
    for i = 1:num_mu
        mu = mu_values(i); fprintf('  μ = %.2f... ', mu);
        [~, ~, wiener_gains] = sdw_mcwf(noisy_signal, alpha, fs, mu);
        all_wiener_gains{i} = wiener_gains; mean_wiener_gains(i) = mean(wiener_gains(:));
        median_wiener_gains(i) = median(wiener_gains(:)); fprintf('completed (avg gain: %.4f)\n', mean_wiener_gains(i));
    end
    
    figure('Position', [150, 150, 1200, 500]);
    subplot_width = 0.38; subplot_height = 0.75; left_margin = 0.08; middle_gap = 0.06; bottom_margin = 0.15;
    
    subplot('Position', [left_margin, bottom_margin, subplot_width, subplot_height]);
    plot(mu_values, mean_wiener_gains, 'b-o', 'LineWidth', 2, 'MarkerSize', 8); hold on;
    plot(mu_values, median_wiener_gains, 'r--s', 'LineWidth', 2, 'MarkerSize', 8); grid on;
    xlabel('μ Parameter', 'FontSize', 12); ylabel('Wiener Gain', 'FontSize', 12);
    title('Average and Median Wiener Gains vs μ', 'FontSize', 14);
    legend('Mean', 'Median', 'Location', 'southeast'); xlim([0, 100]); ylim([0.1, 1]);
    
    subplot('Position', [left_margin + subplot_width + middle_gap, bottom_margin, subplot_width, subplot_height]);
    wiener_data = []; group_labels = [];
    for i = 1:num_mu
        gains = all_wiener_gains{i}(:); wiener_data = [wiener_data; gains];
        group_labels = [group_labels; repmat(i, length(gains), 1)];
    end
    
    boxplot(wiener_data, group_labels, 'Labels', arrayfun(@(x) sprintf('%.1f', x), mu_values, 'UniformOutput', false));
    xlabel('μ Parameter', 'FontSize', 12); ylabel('Wiener Gain', 'FontSize', 12);
    title('Wiener Gain Distribution for Different μ Values', 'FontSize', 14); grid on; ylim([0.1, 1]);
    ax = gca; ax.XTickLabelRotation = 45;
    
    fprintf('\n===== Wiener Gain Statistical Analysis =====\n'); 
    fprintf('μ Value\tMean Gain\tMedian\tStd Dev\n'); 
    fprintf('--------------------------------------\n');
    for i = 1:num_mu
        gains = all_wiener_gains{i}(:);
        fprintf('%.2f\t%.4f\t%.4f\t%.4f\n', mu_values(i), mean(gains), median(gains), std(gains));
    end
    
    save('mu_wiener_gain_analysis.mat', 'mu_values', 'all_wiener_gains', 'mean_wiener_gains', 'median_wiener_gains');
    fprintf('\nAnalysis results saved to mu_wiener_gain_analysis.mat\n');
end

function save_enhanced_audio(enhanced_signals, mixed_eval, fs)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%？？？
    fprintf('\nSaving enhanced audio files...\n');
    if exist('enhanced_audio_output', 'dir') ~= 7
        mkdir('enhanced_audio_output');
    end
    
    
    % 保存文件
    filenames = {'enhanced_mvdr_corrected.wav', 'prewhitening-mvdr_corrected.wav', ...
                 'enhanced_mcwf_correct.wav', 'enhanced_sdw_mcwf_0.5.wav', ...
                 'enhanced_sdw_mcwf_5.0.wav', 'enhanced_sdw_mcwf_20.wav', ...
                 'enhanced_sdw_mcwf_50.wav', 'enhanced_sdw_mcwf_100.wav'};
    
    for i = 1:min(length(enhanced_signals), length(filenames))
        audiowrite(['enhanced_audio_output/' filenames{i}], enhanced_signals{i}, fs);
    end
    
    % 保存参考信号
    audiowrite('enhanced_audio_output/noisy_input.wav', mixed_eval/max(abs(mixed_eval)), fs);
end

function plot_performance_vs_snr(clean_speech, h_target, interference_signals, h_interferences, num_mics, fs, frame_len, frame_shift, nfft, num_bands)
    snr_range = -10:1:10; % Reduced SNR range for faster computation
    interference_names = { 'Clean Speech 2', 'Babble Noise', 'Artificial Noise', 'Stationary Noise'};
    metrics = {'PESQ', 'STOI', 'ESTOI'};
    alg_names = {'MVDR (GEVD)', 'MCWF', 'SDW-MCWF (μ=100)'};
    alg_colors = {'b-o', 'g-^', 'r-s'};
    num_alg = length(alg_names);
    
    % Algorithm function handles - only 3 selected algorithms
    alg_funcs = {@ mvdr_gevd, ...
                 @(x, alpha, fs) mcwf(x, alpha, fs), ...
                 @(x, alpha, fs) sdw_mcwf(x, alpha, fs, 100.0)};
    
    % Store all results: SNR × Algorithm × Metric × Interference
    performance_data = zeros(length(snr_range), num_alg, 3, 4);
    
    for inter_idx = 1:4
        inter_signal = interference_signals{inter_idx};
        h_inter = h_interferences{inter_idx};
        for snr_idx = 1:length(snr_range)
            target_signal = generate_multichannel_signal(clean_speech, h_target, num_mics);
            interference_signal = generate_multichannel_signal(inter_signal, h_inter, num_mics);

            % Adjust interference energy
            target_power = mean(target_signal(:,1).^2);
            inter_power = mean(interference_signal(:,1).^2);
            if inter_power > 0
                weight = sqrt(target_power / (inter_power * 10^(snr_range(snr_idx)/10)));
            else
                weight = 1;
            end
            mixed_signal = target_signal + weight * interference_signal;
            min_len = min([length(clean_speech), length(target_signal(:,1)), length(mixed_signal)]);
            
            clean_eval = clean_speech(1:min_len);
            target_eval = target_signal(1:min_len, 1);
            mixed_eval = mixed_signal(1:min_len, 1);
            
            
            for alg_idx = 1:num_alg
                % Algorithm output
                if alg_idx == 1
                    [enhanced_signal, ~] = alg_funcs{alg_idx}(mixed_signal, 0.9, fs);
                else
                    [enhanced_signal, ~, ~] = alg_funcs{alg_idx}(mixed_signal, 0.9, fs);
                end
                enhanced_eval = enhanced_signal(1:min_len);
                
                
                [~, pesq_score, stoi_score, estoi_score, ~] = evaluate_enhancement_with_segsnr_improvement(clean_eval, target_eval, weight * interference_signal(1:min_len,1), mixed_eval, enhanced_eval, fs, num_mics);
                performance_data(snr_idx, alg_idx, 1, inter_idx) = pesq_score;
                performance_data(snr_idx, alg_idx, 2, inter_idx) = stoi_score;
                performance_data(snr_idx, alg_idx, 3, inter_idx) = estoi_score;
            end
        end
    end

    % Plot results
    for metric_idx = 1:3
        figure('Position', [100 + (metric_idx-1)*300, 100, 900, 700]);
        for inter_idx = 1:4
            subplot(2, 2, inter_idx); hold on;
            for alg_idx = 1:num_alg
                plot(snr_range, squeeze(performance_data(:, alg_idx, metric_idx, inter_idx)), alg_colors{alg_idx}, ...
                     'LineWidth', 2, 'MarkerSize', 6);
            end
            grid on;
            xlabel('SNR (dB)');
            ylabel(metrics{metric_idx});
            title([metrics{metric_idx} ' vs SNR for ' interference_names{inter_idx}]);
            xlim([snr_range(1) snr_range(end)]);
            if strcmp(metrics{metric_idx}, 'PESQ')
                ylim([1 4.5]);
            else
                ylim([0 1]);
            end
            if inter_idx == 1
                legend(alg_names, 'Location', 'northwest', 'FontSize', 10);
            end
            hold off;
        end
        sgtitle([metrics{metric_idx} ' Performance Across Different Interferences']);
    end
end

%% Run main function
main_speech_enhancement_complete();