package com.pets.pets_service.Registration.token;

import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.Optional;

@Service
@AllArgsConstructor
public class PPConfirmationTokenService {

    private final PPConfirmationTokenRepository confirmationTokenRepository;

    public void saveConfirmationToken(PPConfirmationToken token) {
        confirmationTokenRepository.save(token);
    }

    public Optional<PPConfirmationToken> getToken(String token) {
        return confirmationTokenRepository.findByToken(token);
    }

    public int setConfirmedAt(String token) {
        return confirmationTokenRepository.updateConfirmedAt(
                token, LocalDateTime.now());
    }
}
