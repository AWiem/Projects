package com.pets.pets_service.Registration.token;

import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.Optional;

@Service
@AllArgsConstructor
public class ClientConfirmationTokenService {

    private final ClientConfirmationTokenRepository confirmationTokenRepository;

    public void saveConfirmationToken(ClientConfirmationToken token) {
        confirmationTokenRepository.save(token);
    }

    public Optional<ClientConfirmationToken> getToken(String token) {
        return confirmationTokenRepository.findByToken(token);
    }

    public int setConfirmedAt(String token) {
        return confirmationTokenRepository.updateConfirmedAt(
                token, LocalDateTime.now());
    }
}
