package com.pets.pets_service.Registration.token;


import com.pets.pets_service.Models.ProductProvider;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.persistence.*;
import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@Entity
public class PPConfirmationToken {

    @SequenceGenerator(
        name = "PP_confirmation_token_sequence",
        sequenceName = "PP_confirmation_token_sequence",
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
    private ProductProvider productprovider;

    public PPConfirmationToken(String token,
                             LocalDateTime createdAt,
                             LocalDateTime expiresAt,
                             ProductProvider productprovider) {
        this.token = token;
        this.createdAt = createdAt;
        this.expiresAt = expiresAt;
        this.productprovider = productprovider;
    }
}
