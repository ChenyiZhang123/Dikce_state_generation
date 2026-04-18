clear
clc
tic

% ========================================================================
% Dicke state generation via generalized parity measurement (GPM)
% ------------------------------------------------------------------------
% This script numerically simulates the Dicke-state generation protocol
% discussed in the paper
% "Efficient nonclassical state preparation via generalized parity
% measurement".
%
% Physical idea:
%   - A collective spin-J system (dimension 2J+1) is used to represent the
%     symmetric subspace of a spin ensemble.
%   - Repeated nonunitary filtering operations are applied to the state.
%   - Each round approximately removes Dicke components that violate a
%     certain generalized parity condition.
%   - As the rounds proceed, the state is driven toward the target Dicke
%     state |J, m_t>.
%
% In this code, the target is the center Dicke state with m_t = 0.
%
% Main outputs:
%   Pn(ii)     : fidelity with the target Dicke state after round ii
%   Pg(ii+1)   : cumulative success probability after round ii
%   Fisher(ii) : quantum Fisher information (QFI) for phase sensing around
%                the x-axis after round ii
%
% Reference:
%   C.-y. Zhang and J. Jing,
%   Phys. Rev. A 113, 022420 (2026).
% ========================================================================

NN = 10;                 % Number of evolution-measurement rounds

%% Parameters of the collective spin system
J = 50;                  % Total angular momentum quantum number
ParN = 2*J;              % Total number of physical spin-1/2 particles, M = 2J
nn = 2*J + 1;            % Dimension of the symmetric Dicke subspace

% Magnetic quantum numbers m = -J, -J+1, ..., J
m = linspace(-J, J, nn);

% Matrix elements for the collective ladder operators in the Dicke basis:
% J_+ |J,m> = sqrt[J(J+1)-m(m+1)] |J,m+1>
% J_- |J,m> = sqrt[J(J+1)-m(m-1)] |J,m-1>
%
% Here the coefficient array 'n' is arranged for building J_- using spdiags.
n = sqrt(J*(J+1)*ones(1,nn) - m.*(m-ones(1,nn)));

%% Construct collective spin operators in the Dicke basis
Jm = full(spdiags(n', 1, nn, nn));   % Lowering operator J_-
Jp = Jm';                            % Raising operator J_+
Jx = (Jp + Jm)/2;                    % Collective spin operator J_x
Jy = (Jp - Jm)/(2*1i);               % Collective spin operator J_y

%% Initial state preparation
% Initialize the state vector in the Dicke basis.
% Basis ordering here is |J,-J>, |J,-J+1>, ..., |J,J>.
inic = zeros(nn,1);

% ------------------------------------------------------------------------
% Option 1 (commented out in the original code):
% Prepare the spin coherent state pointing along +x directly from the
% binomial coefficients in the Dicke basis.
%
% for kk=-J:1:J
%     inic(kk+J+1)=sqrt(nchoosek(ParN, J+kk)/2^ParN);
% end
% ------------------------------------------------------------------------

% Option 2 (used here):
% Start from the fully polarized Dicke state |J,-J>, then rotate it by
% exp(i Jy pi/2). This generates a spin coherent state approximately
% aligned along the x direction.
inic(1) = 1;
inic = expm(1i*Jy*pi/2) * inic;

%% Target Dicke state |J, m_t>
mt = 0;                              % Target magnetic quantum number

target = zeros(nn,1);
target(mt + J + 1) = 1;              % Target state vector in Dicke basis

%% Arrays for storing results
Pn = zeros(1,NN);                    % Fidelity with target state after each round
Pg = zeros(1,NN+1);                  % Cumulative success probability
Fisher = zeros(1,NN);                % Quantum Fisher information after each round
Pt_Dicke = zeros(1,NN);              % Reserved array (currently unused)

Pg(1) = 1;                           % Before any measurement, success probability = 1
ket = inic;                          % Current normalized state of the spin ensemble

%% Nonunitary filtering operators for Dicke-state generation
% The paper uses two kinds of effective nonunitary evolution operators,
% depending on whether the ancillary spin is measured in |e> or |g>.
%
% For the center Dicke state |J,0>, the first round uses one branch and
% the subsequent rounds use the other branch. The two diagonal operators
% below encode the corresponding coefficients in the Dicke basis.
%
% Vlee_dicke(l): first-round filtering coefficients
% Vlgg_dicke(l): later-round filtering coefficients
%
% Each operator is diagonal in the Dicke basis, so applying it amounts to
% weighting each |J,m> component by a cosine factor.

Vlee_dicke = @(l) ( ...
    cos(l*pi*sqrt((J*(J+1)*ones(1,2*J+1) - m.*(m+ones(1,2*J+1))) ./ ...
                  ((J+1)*J*ones(1,2*J+1) - mt*(mt+1)))) ...
    );

Vlgg_dicke = @(l) ( ...
    cos(l*pi*sqrt((J*(J+1)*ones(1,2*J+1) - m.*(m-ones(1,2*J+1))) ./ ...
                  ((J+1)*J*ones(1,2*J+1) - mt*(mt-1)))) ...
    );

%% Choice of discrete evolution-time parameters
% The protocol uses stepwise halved effective times.
% half_list = [1/2, 1/4, 1/8, ..., 2^{-NN}]
% These are then rescaled and rounded to obtain integer parameters l_k.
half_list = 2.^(-linspace(1, NN, NN));   % Suitable for even particle number M = 2J

% Alternative choice used in some cases with odd particle number:
% half_list = 2.^(-linspace(-2, NN-3, NN));

%% Main iteration: repeated evolution + measurement
for ii = 1:NN

    % For this parameter choice, l_list is recomputed each round.
    % It follows the halving strategy discussed in the paper.
    l_list = round((J*(J+1))^(1) * half_list);

    if ii < 2
        % First round: use the operator associated with the first filtering
        % step, which removes the undesired component near |J,1>.
        V = diag(Vlee_dicke(l_list(ii)));
    else
        % Later rounds: use the operator for repeated generalized parity
        % filtering toward the target Dicke state.
        V = diag(Vlgg_dicke(l_list(ii-1)));
    end

    % Apply the nonunitary operator.
    kettemp = V * ket;

    % Renormalize the state conditional on successful measurement outcome.
    ket = kettemp / sqrt(kettemp' * kettemp);

    % Fidelity with the target Dicke state after this round:
    % Pn(ii) = |<target|ket>|^2
    Pn(ii) = abs(ket' * target)^2;

    % Update cumulative success probability.
    % (kettemp' * kettemp) is the conditional success probability for this
    % round given all previous successful rounds.
    Pg(ii+1) = (kettemp' * kettemp) * Pg(ii);

    % Quantum Fisher information for estimating a phase generated by J_x:
    % F_Q = 4(<J_x^2> - <J_x>^2)
    Fisher(ii) = 4*(ket' * Jx^2 * ket - abs(ket' * Jx * ket)^2);
end

%% Plot results
figure(1)

subplot(2,1,1)
plot(1:NN, Pn(1:NN), 'Marker', 'o', 'MarkerIndices', 1:NN), hold on
xlabel('Round number N')
ylabel('Fidelity')
title('Target Dicke-state fidelity')

subplot(2,1,2)
plot(1:NN, Pg(2:NN+1), 'Marker', 'o', 'MarkerIndices', 1:NN), hold on
xlabel('Round number N')
ylabel('Success probability')
title('Cumulative success probability')

% ------------------------------------------------------------------------
% Optional diagnostics
% ------------------------------------------------------------------------
% figure(2)
% bar(-J:1:J, diag(ket*ket'))
% xlim([-20 20])
% xlabel('m')
% ylabel('Population')
% title('Final Dicke-basis population distribution')
%
% figure(4)
% plot(1:NN, Fisher(1:NN), 'Marker', 'o', 'MarkerIndices', 1:NN), hold on
% xlabel('Round number N')
% ylabel('Quantum Fisher information')
% title('QFI during Dicke-state generation')

%% End timing
toc
