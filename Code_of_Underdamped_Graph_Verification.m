% ================================================================
% Line-Following Robot — Underdamped Case Only
% Gabriel Melgarejo — Math 2A
% ================================================================
% IVP: e''(t) + 2e'(t) + 4e(t) = 0
%      e(0) = 0.1, e'(0) = 0
% ================================================================

clc; clear; close all;

% ----------------------------------------------------------------
% PARAMETERS
% ----------------------------------------------------------------
m  = 1.0;   % mass of robot        [kg]
kp = 4.0;   % proportional gain    [N/m]
kd = 2.0;   % derivative gain      [N·s/m] — underdamped case
e0 = 0.1;   % initial error        [m]
de0 = 0;    % initial velocity     [m/s]

% ----------------------------------------------------------------
% ANALYTICAL SOLUTION
% e(t) = e^(-t)(0.1*cos(sqrt(3)*t) + (0.1/sqrt(3))*sin(sqrt(3)*t))
% ----------------------------------------------------------------
t_analytical = linspace(0, 8, 1000);
alpha = -1;
beta  = sqrt(3);
C1    = 0.1;
C2    = 0.1 / sqrt(3);
e_analytical = exp(alpha*t_analytical) .* ...
               (C1*cos(beta*t_analytical) + ...
                C2*sin(beta*t_analytical));

% ----------------------------------------------------------------
% PREDICTION AT t = 2
% ----------------------------------------------------------------
t_pred = 2;
e_pred = exp(alpha*t_pred) * ...
         (C1*cos(beta*t_pred) + C2*sin(beta*t_pred));

fprintf('\n--- Prediction at t = 2 seconds ---\n');
fprintf('e(2) = %.4f meters\n', e_pred);
fprintf('Robot is %.2f cm on the opposite side of the line\n', abs(e_pred*100));

% ----------------------------------------------------------------
% PLOT
% ----------------------------------------------------------------
figure(1);
hold on;

% Analytical solution curve
plot(t_analytical, e_analytical, 'b-', 'LineWidth', 2.5, ...
    'DisplayName', 'Underdamped solution $e(t)$');

% Target line
yline(0, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2, ...
    'DisplayName', 'Target: $e(t) = 0$');

% Prediction point
plot(t_pred, e_pred, 'o', 'MarkerSize', 10, ...
    'MarkerFaceColor', 'yellow', ...
    'MarkerEdgeColor', 'black', ...
    'LineWidth', 1.5, ...
    'DisplayName', sprintf('Prediction: $e(2) \\approx %.4f$ m', e_pred));

% Dashed lines to axes for the prediction point
xline(t_pred, ':', 'Color', [1 0.6 0], 'LineWidth', 1.2, ...
    'HandleVisibility', 'off');
yline(e_pred, ':', 'Color', [1 0.6 0], 'LineWidth', 1.2, ...
    'HandleVisibility', 'off');

% ----------------------------------------------------------------
% FORMATTING
% ----------------------------------------------------------------
xlabel('Time $t$ (seconds)', 'FontSize', 13, 'Interpreter', 'latex');
ylabel('Lateral Error $e(t)$ (meters)', 'FontSize', 13, 'Interpreter', 'latex');
title('Underdamped Lateral Error Correction with Prediction', ...
    'FontSize', 14, 'Interpreter', 'latex');
legend('Location', 'best', 'FontSize', 11, 'Interpreter', 'latex');
grid on;
xlim([0 8]);
ylim([-0.08 0.12]);
hold off;

% ----------------------------------------------------------------
% SAVE
% ----------------------------------------------------------------
exportgraphics(gcf, 'underdamped_prediction.png', 'Resolution', 300);
fprintf('Graph saved as underdamped_prediction.png\n');