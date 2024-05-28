package com.pets.pets_service.Registration.token;


import com.pets.pets_service.Models.Veterinary;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.persistence.*;
import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@Entity
public class VetConfirmationToken {

    @SequenceGenerator(
        name = "vet_confirmation_token_sequence",
        sequenceName = "vet_confirmation_token_sequence",
        allocationSize = 1
    )
    @Id
    @GeneratedValue(
            strategy = GenerationType.SEQUENCE,
            generator = "confirmation_token_sequence"
    )
    private Long id;

    @Column(nullable = false)
    private String token;

    @Column(nullable = false)
    private LocalDateTime createdAt;

    @Column(nullable = false)
    private LocalDateTime expiresAt;

    private LocalDateTime confirmedAt;

    @ManyToOne
    @JoinColumn(
            nullable = false,
            name = "app_user_id"
    )
    private Veterinary vet;

    public VetConfirmationToken(String token,
                             LocalDateTime createdAt,
                             LocalDateTime expiresAt,
                             Veterinary vet) {
        this.token = token;
        this.createdAt = createdAt;
        this.expiresAt = expiresAt;
        this.vet = vet;
    }
}
